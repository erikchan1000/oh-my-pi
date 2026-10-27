/**
 * `omp --restore`: a session goes on with what its last process left unfinished when
 * it ended ({@link restoreSession}): the turn it was in
 * ({@link AgentSession.restoreInterruptedTurn}) and the subagents it was running.
 *
 * Every persisted subagent transcript under the session ({@link collectSubSessions})
 * that its process left mid-turn ({@link hasRestorableTurn}), was not killed (no
 * tombstone) and can be revived (a `session_init` contract; an isolated one re-adopts its surviving sandbox from its sidecar, else it is reported as unrestorable with the reason) runs one
 * more monitored turn as a background job of its parent: the lifecycle reviver
 * rebuilds it from its transcript, {@link runSubagentFollowUpTurn} goes on with the
 * cut-off turn and drives it to `yield`, and the parent receives the result as the
 * ordinary `async-result` of a job with the subagent's id, the id its first run's
 * job had, so a parent waiting on that job finds it again.
 */
import * as path from "node:path";
import { logger, prompt } from "@oh-my-pi/pi-utils";
import { AsyncJobError } from "../async";
import { AgentLifecycleManager } from "../registry/agent-lifecycle";
import { ensurePersistedRoster } from "../registry/persisted-agents";
import { AgentRegistry } from "../registry/agent-registry";
import type { AgentSession } from "../session/agent-session";
import { hasRestorableTurn } from "../session/exit-diagnostics";
import { extractSessionInit } from "../session/session-manager";
import { collectSubSessions } from "../session/sub-sessions";
import type { EventBus } from "../utils/event-bus";
import { runSubagentFollowUpTurn } from "./executor";
import { persistedAgentDefinition } from "./persisted-revive";
import isolationWorkspaceKeptTemplate from "../prompts/tools/isolation-workspace-kept.md" with { type: "text" };
import { finishAdoptedIsolation } from "./isolation-runner";
import { formatTaskResultSummary } from "./result-summary";
import { cfgTaskMaxRuntimeMs } from "./settings";
import {
	abandonAdoptedSandbox,
	type AdoptIsolatedSandboxResult,
	adoptIsolatedSandbox,
	forgetAdoptedSandbox,
} from "./worktree";

/** Where restored subagents' lifecycle and progress frames go, as a live run's do. */
export interface RestoreOptions {
	eventBus?: EventBus;
	subagentEventBus?: EventBus;
}

/** What `omp --restore` went on with. */
export interface RestoreOutcome {
	/** The session's own interrupted turn goes on. */
	turn: boolean;
	/** Ids of the subagents that go on with theirs, parents before their children. */
	subagents: string[];
	/** Interrupted subagents that cannot go on, each with the specific reason. */
	unrestorable: Array<{ id: string; reason: string }>;
}

interface RestoredSubagents {
	restored: string[];
	unrestorable: Array<{ id: string; reason: string }>;
}

/**
 * Has `session` go on with what its last process left unfinished: its subagents'
 * interrupted turns, each as a background job of its parent (registered first, so
 * a resumed turn waiting on such a job finds it), then its own interrupted turn.
 */
export async function restoreSession(session: AgentSession, options: RestoreOptions = {}): Promise<RestoreOutcome> {
	const { restored, unrestorable } = await restoreSubagents(session, options);
	const turn = await session.restoreInterruptedTurn();
	return { turn, subagents: restored, unrestorable };
}

/**
 * Has every subagent of `session` that its last process left mid-turn go on with that
 * turn (see the module docs). Nothing is restored for a session without a file, or
 * without background jobs (the `async.enabled` setting off), which could not deliver
 * the results.
 */
async function restoreSubagents(session: AgentSession, options: RestoreOptions): Promise<RestoredSubagents> {
	const sessionFile = session.sessionManager.getSessionFile();
	const manager = session.asyncJobManager;
	if (!sessionFile || !manager) return { restored: [], unrestorable: [] };
	const registry = AgentRegistry.global();
	await ensurePersistedRoster(registry, sessionFile);
	const subSessions = await collectSubSessions(sessionFile, key => {
		logger.warn("Subagent transcript unreadable; not restored", { key });
	});
	// Parents first: a child's result is delivered to its parent's session.
	const keys = Object.keys(subSessions).sort((a, b) => a.split("/").length - b.split("/").length);
	const restored: string[] = [];
	const unrestorable: Array<{ id: string; reason: string }> = [];
	for (const key of keys) {
		const sub = subSessions[key]!;
		if (sub.aborted || !hasRestorableTurn(sub.entries)) continue;
		const id = sub.agentId;
		const ref = registry.get(id);
		const init = extractSessionInit(sub.entries);
		if (!ref?.sessionFile || ref.kind !== "sub" || !init) {
			const reason = "no persisted subagent contract (missing registry ref, session file or session_init)";
			logger.warn("Interrupted subagent cannot be revived; not restored", { id, reason });
			unrestorable.push({ id, reason });
			continue;
		}
		const artifactsDir = path.dirname(ref.sessionFile);
		let adopted: Extract<AdoptIsolatedSandboxResult, { ok: true }> | undefined;
		if (init.isolated) {
			const outcome = await adoptIsolatedSandbox(artifactsDir, id);
			if (!outcome.ok) {
				logger.warn("Interrupted isolated subagent cannot be re-adopted; not restored", {
					id,
					reason: outcome.reason,
				});
				unrestorable.push({ id, reason: outcome.reason });
				continue;
			}
			adopted = outcome;
		}
		const structured = init.tools.includes("yield");
		try {
			manager.register(
				"task",
				id,
				async ({ signal }) => {
					try {
						const result = await runSubagentFollowUpTurn({
							id,
							agent: persistedAgentDefinition(ref.displayName, init),
							message: init.task,
							restore: true,
							structured,
							workPoolYieldItems: init.workPoolYieldItems,
							outputSchema: init.outputSchema,
							outputSchemaMode: init.outputSchemaMode,
							signal,
							eventBus: options.eventBus,
							subagentEventBus: options.subagentEventBus,
							artifactsDir,
							maxRuntimeMs: cfgTaskMaxRuntimeMs.get(session.settings),
						});
						const succeeded = result.exitCode === 0 && !result.aborted && result.error === undefined;
						// A failed or aborted turn leaves the sandbox untouched for manual recovery.
						if (adopted && !succeeded) {
							const failure = formatTaskResultSummary(result, { totalDurationMs: result.durationMs });
							throw new AsyncJobError(
								prompt.render(isolationWorkspaceKeptTemplate, {
									failure,
									workspacePath: adopted.sidecar.workspacePath,
								}),
								result.structuredOutput,
							);
						}
						const finished = adopted
							? await finishAdoptedIsolation({
									sidecar: adopted.sidecar,
									handle: adopted.handle,
									result,
									artifactsDir,
								})
							: result;
						const text = formatTaskResultSummary(finished, { totalDurationMs: finished.durationMs });
						const structuredOutput = finished.structuredOutput;
						if (finished.aborted || finished.exitCode !== 0 || finished.error !== undefined) {
							throw new AsyncJobError(text, structuredOutput);
						}
						return structuredOutput ? { text, structured: structuredOutput } : text;
					} finally {
						if (adopted) {
							// The restored turn settled: the sandbox is merged and gone, or kept for manual
							// recovery. Either way nothing may revive the agent into it, so drop the adoption
							// and end the agent's lifecycle (detaches its session and cached reviver).
							forgetAdoptedSandbox(artifactsDir, id);
							await AgentLifecycleManager.global().release(id);
						}
					}
				},
				{ id, agentId: id, ownerId: ref.parentId },
			);
		} catch (error) {
			// Registration failed after the claim: hand the sandbox back as it was found.
			if (adopted) await abandonAdoptedSandbox(artifactsDir, id);
			throw error;
		}
		restored.push(id);
	}
	if (restored.length > 0) logger.info("Restored interrupted subagents", { ids: restored });
	return { restored, unrestorable };
}
