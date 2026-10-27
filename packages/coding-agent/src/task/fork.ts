import type { Model } from "@oh-my-pi/pi-ai";
import { prompt } from "@oh-my-pi/pi-utils";
import taskForkRefusalTemplate from "../prompts/tools/task-fork-refusal.md" with { type: "text" };
import taskForkSeedTemplate from "../prompts/tools/task-fork-seed.md" with { type: "text" };
import { collectPendingToolCalls, createInterruptedToolResults } from "../session/exit-diagnostics";
import type { SessionEntry } from "../session/session-entries";
import { SessionManager } from "../session/session-manager";
import type { ConfiguredThinkingLevel } from "@oh-my-pi/pi-tui/thinking";

/** Custom session entry marking a transcript as a forked-context task child. */
export const TASK_FORK_ENTRY_TYPE = "task_fork";

/** Payload of {@link TASK_FORK_ENTRY_TYPE}: the pending calls inherited in the child's prefix. */
export interface TaskForkEntryData {
	forkedFromToolCallId: string;
	/** Pending tool call ids from the parent turn (this fork's own call plus its siblings) and earlier fork ancestors. */
	ancestorPendingToolCallIds: string[];
	/** `task` texts of the pending `task` calls above; a fork child repeating one is told to focus on its own task. */
	ancestorPendingTasks: string[];
}

/** Everything the executor needs to branch the live parent for one fork spawn. */
export interface ForkSpawnSpec {
	sourceFile: string;
	/** Parent leaf entry at the `task` call; the child branches here even if the parent has since moved on. */
	leafId: string;
	/** Parent base system prompt, reused byte for byte. */
	systemPrompt: readonly string[];
	model: Model;
	thinkingLevel: ConfiguredThinkingLevel | undefined;
	/** Parent's enabled tool names; the child must end up with the same set. */
	toolNames: string[];
	/** Parent's xd://-mounted tools: advertised through `write xd://`, not as direct provider tools. */
	mountedToolNames: string[];
	promptCacheKey: string;
	/** Parent's spawn policy; the child's task tool description renders from it, so it must match. */
	spawns: string;
	/** Id of the pending `task` call the child's seed result answers. */
	toolCallId: string;
	entry: TaskForkEntryData;
}

/** Per-spawn params carrying the dispatch-time fork snapshot. */
export interface ForkAwareTaskParams {
	forkSpec?: ForkSpawnSpec;
}

export type ForkRefusal =
	| { reason: "disabled" }
	| { reason: "noSessionFile" }
	| { reason: "noAgentId" }
	| { reason: "noParent" }
	| { reason: "noUsage" }
	| { reason: "tooLarge"; tokens: number; maxTokens: number }
	| { reason: "missingState" }
	| { reason: "unsupportedTools"; tools: string }
	| { reason: "missingTools"; tools: string }
	| { reason: "toolDrift"; childTools: string; parentTools: string }
	| { reason: "legacyFork"; id: string }
	| { reason: "conflict"; field: string };

/** Render an explicit, model-facing refusal for a `fork: true` call. */
export function renderForkRefusal(refusal: ForkRefusal): string {
	return prompt.render(taskForkRefusalTemplate, { ...refusal, [refusal.reason]: true }).trim();
}

/** Render the synthetic tool result answering the pending `task` call in the child's prefix. */
export function renderForkSeed(id: string, task: string): string {
	return prompt.render(taskForkSeedTemplate, { id, task }).trim();
}

function taskTextsOf(args: unknown): string[] {
	if (args === null || typeof args !== "object") return [];
	const record = args as { task?: unknown; tasks?: unknown };
	const texts: string[] = [];
	if (typeof record.task === "string") texts.push(record.task.trim());
	if (Array.isArray(record.tasks)) {
		for (const item of record.tasks) {
			if (item !== null && typeof item === "object" && typeof (item as { task?: unknown }).task === "string") {
				texts.push((item as { task: string }).task.trim());
			}
		}
	}
	return texts;
}

/** Read every `task_fork` entry on a branch (the child's own plus those inherited from forked ancestors). */
export function readForkEntries(entries: readonly SessionEntry[]): TaskForkEntryData[] {
	return entries
		.filter(entry => entry.type === "custom" && entry.customType === TASK_FORK_ENTRY_TYPE)
		.map(entry => (entry as { data: TaskForkEntryData }).data);
}

/** Build the entry recorded in a new fork child: this turn's pending calls plus inherited fork ancestry. */
export function buildForkEntry(branch: readonly SessionEntry[], forkedFromToolCallId: string): TaskForkEntryData {
	const pending = collectPendingToolCalls(branch).map(call => ({
		id: call.toolCallId,
		tasks: call.toolName === "task" ? taskTextsOf(call.args) : [],
	}));
	const inherited = readForkEntries(branch);
	return {
		forkedFromToolCallId,
		ancestorPendingToolCallIds: [
			...new Set([
				...pending.flatMap(call => (call.id === undefined ? [] : [call.id])),
				...inherited.flatMap(e => e.ancestorPendingToolCallIds),
			]),
		],
		ancestorPendingTasks: [
			...new Set([...pending.flatMap(call => call.tasks), ...inherited.flatMap(e => e.ancestorPendingTasks)]),
		],
	};
}

export type ForkGuardVerdict = { kind: "allow" } | { kind: "replay" } | { kind: "depth" };

/**
 * Replay guard for a fork child's own `task` calls. A call repeating an ancestor-pending task text is the
 * child confusing itself with the parent; any other spawn needs recursion headroom.
 */
export function checkForkGuard(
	entries: readonly SessionEntry[],
	taskTexts: readonly string[],
	depthAllowsSpawning: boolean,
): ForkGuardVerdict {
	const forks = readForkEntries(entries);
	if (forks.length === 0) return { kind: "allow" };
	const ancestorTasks = new Set(forks.flatMap(entry => entry.ancestorPendingTasks));
	if (taskTexts.some(text => ancestorTasks.has(text.trim()))) return { kind: "replay" };
	return depthAllowsSpawning ? { kind: "allow" } : { kind: "depth" };
}

/**
 * Branch the live parent into a writable child transcript. The parent's pending tool calls get
 * results (the fork's own call gets the seed, siblings an unknown-outcome note), so the child's
 * first request is the parent prefix followed by the seed tool result.
 */
export async function createForkChildManager(
	spec: ForkSpawnSpec,
	cwd: string,
	sessionFile: string | undefined,
	childId: string,
	task: string,
): Promise<SessionManager> {
	if (sessionFile === undefined) throw new Error("A forked task requires a persisted child session file.");
	const manager = await SessionManager.forkFrom(spec.sourceFile, cwd, undefined, undefined, {
		copyArtifacts: false,
		suppressBreadcrumb: true,
		sessionFile,
		resetInheritedCost: true,
	});
	manager.branch(spec.leafId);
	let seeded = false;
	for (const result of createInterruptedToolResults(manager.getBranch(), "fork")) {
		if (result.toolCallId === spec.toolCallId) {
			seeded = true;
			manager.appendMessage({ ...result, content: [{ type: "text", text: renderForkSeed(childId, task) }] });
		} else {
			manager.appendMessage(result);
		}
	}
	if (!seeded) {
		throw new Error(`Forked task: pending task call ${spec.toolCallId} is not in the parent's persisted transcript.`);
	}
	manager.appendCustomEntry(TASK_FORK_ENTRY_TYPE, spec.entry);
	return manager;
}
