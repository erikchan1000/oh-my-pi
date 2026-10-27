/**
 * Reusable isolation lifecycle for subagent execution.
 *
 * Both `TaskTool` and the eval `agent()` bridge spawn subagents that can run
 * inside a copy-on-write worktree, capture their changes, and (optionally)
 * apply those changes back to the parent repo. The orchestration is identical
 * for both callers; this module hosts the shared lifecycle so eval `agent()`
 * does not need to round-trip through `TaskTool.#runSpawn`.
 *
 * Shape:
 *   1. {@link prepareIsolationContext} — resolve the git root.
 *   2. {@link runIsolatedSubprocess}    — start worktree, capture its
 *                                        baseline, run, capture changes, and
 *                                        transfer cleanup ownership.
 *   3. {@link mergeIsolatedChanges}     — apply captured changes back to the
 *                                        parent repo (skip when the caller
 *                                        opted out).
 *
 * Step 1 happens once per top-level call; steps 2 and 3 are per-spawn.
 */
import * as fs from "node:fs/promises";
import * as path from "node:path";
import type * as natives from "@oh-my-pi/pi-natives";
import * as vcs from "@oh-my-pi/pi-natives/vcs";
import { prompt } from "@oh-my-pi/pi-utils";
import isolationErrorTemplate from "../prompts/tools/isolation-error.md" with { type: "text" };
import isolationSummaryTemplate from "../prompts/tools/isolation-summary.md" with { type: "text" };
import { AgentLifecycleManager } from "../registry/agent-lifecycle";
import { AgentRegistry } from "../registry/agent-registry";
import type { ToolSession } from "../tools";
import { generateCommitMessage } from "../utils/commit-message-generator";
import { trackLateCleanup } from "../utils/late-cleanup";
import { withRepoLock } from "../utils/repo-lock";
import type { ExecutorOptions } from "./executor";
import { runSubprocess } from "./executor";
import { needsNativeTeardown, writeRetainedBackend } from "./isolation-ownership";
import type { NestedRepoPatch, SingleResult } from "@oh-my-pi/pi-tui/tools/task";
import {
	applyNestedPatches,
	captureDeltaPatch,
	captureIsolationBaseline,
	cleanupIsolation,
	cleanupTaskBranches,
	type CommitToBranchResult,
	commitToBranch,
	ensureIsolation,
	getRepoRoot,
	type IsolationHandle,
	type IsolationSidecar,
	applyPatchWithThreeWayFallback,
	mergeTaskBranches,
	patchTouchedFiles,
	probePatchApplies,
	removeIsolationSidecar,
	type WorktreeBaseline,
	writeIsolationSidecar,
} from "./worktree";

import { cfgTaskIsolationCommits } from "./settings";

type IsoBackendKind = natives.IsoBackendKind;

/** Which isolation outcome `isolation-summary.md` should describe. */
export type IsolationSummaryKind =
	| "captured"
	| "capture-error"
	| "nested-apply-failed"
	| "not-applied"
	| "patch-conflict"
	| "branch-merge-failed"
	| "branch-capture-failed"
	| "merge-error";

/** Context for `isolation-summary.md`; unused fields are simply absent. */
export interface IsolationSummaryContext {
	kind: IsolationSummaryKind;
	branchName?: string;
	/** Root patch path, only when it holds changes. */
	rootPatchPath?: string;
	nestedCount?: number;
	nestedPatchPaths?: string[];
	error?: string;
	conflict?: string;
	/** Paths whose hunks conflict with the current parent tree (`patch-conflict`). */
	conflictPaths?: string[];
}

/**
 * Render one isolation outcome as the model-facing suffix appended to a task
 * result. Always starts with a blank line so it separates from the output it
 * follows.
 */
export function renderIsolationSummary(context: IsolationSummaryContext): string {
	return `\n\n${prompt.render(isolationSummaryTemplate, { ...context })}`;
}

/** Record artifact locations for `agent://` and mark the result as an isolated run. */
function rememberAgentArtifacts(result: SingleResult): SingleResult {
	AgentRegistry.global().setHistory(result.id, {
		outputPath: result.outputPath,
		patchPath: result.patchPath,
		branchName: result.branchName,
		nestedPatchPaths: result.nestedPatchPaths,
	});
	return { ...result, isolated: true };
}

/**
 * Decide the fate of a half-built task branch after apply-back threw.
 *
 * Returns the branch name when it carries at least one commit past `baseSha`
 * — the caller must keep it, because the isolation worktree that also held
 * those objects is about to be torn down. Returns `undefined` after deleting
 * a branch that is absent, empty, or still pinned at the baseline, preserving
 * the original stale-branch cleanup for the cases where nothing is at stake.
 *
 * `revList.range` throws when the branch does not exist, which is the common
 * "commitToBranch failed before it created anything" path; that is treated as
 * "nothing to rescue" only after confirming the ref is absent. Other probe
 * failures preserve the branch because deleting it could lose the only reachable
 * copy of the agent's commits.
 */
async function rescueTaskBranch(repoRoot: string, branchName: string, baseSha: string): Promise<string | undefined> {
	const repo = vcs.git(repoRoot);
	try {
		const carriedCommits = (await vcs.requireGit(repoRoot).revListRange(baseSha, branchName)).length;
		if (carriedCommits > 0) return branchName;
	} catch {
		try {
			if (await repo?.refExists(`refs/heads/${branchName}`)) return branchName;
		} catch {
			// An inconclusive recovery probe must never risk deleting the only ref.
			return branchName;
		}
	}
	try {
		await repo?.deleteBranch(branchName, true);
	} catch {
		// Best-effort cleanup matches the old façade's tryDelete semantics.
	}
	return undefined;
}

/** Resolved repo shared by every isolated spawn in a single call. */
export interface IsolationContext {
	repoRoot: string;
}

/**
 * Resolve the git repo root isolated spawns are copied from and merged back
 * into. Throws when the cwd is not inside a git repository; callers surface
 * the error as a task-tool failure. Each spawn captures its own baseline from
 * its isolation copy (see {@link captureIsolationBaseline}).
 */
export async function prepareIsolationContext(cwd: string): Promise<IsolationContext> {
	return { repoRoot: await getRepoRoot(cwd) };
}

/** Fingerprint of a captured delta, used to skip re-committing an unchanged workspace on release. */
function deltaFingerprint(rootPatch: string, nestedPatches: readonly NestedRepoPatch[]): bigint | number {
	let key = rootPatch;
	for (const { relativePath, patch } of nestedPatches) key += `\0${relativePath}\0${patch}`;
	return Bun.hash(key);
}

/** Build a commit-message callback for branch/nested commits; `undefined` ⇒ fall back to generic message. */
export type BuildCommitMessage = () => undefined | ((diff: string) => Promise<string | null>);

/**
 * Construct the commit-message factory used by isolation branch commits and
 * nested-repo patch commits. Returns a closure that, each time it's called,
 * either yields an AI-backed `(diff) => Promise<string|null>` callback (when
 * `task.isolation.commits === "ai"` and a model registry is available) or
 * `undefined` so the caller falls back to a generic commit message.
 *
 * Centralized so `TaskTool` and the eval `agent()` bridge share one wiring;
 * a drift here previously meant the two callers built subtly different
 * generators for the same setting.
 */
export function makeIsolationCommitMessage(session: ToolSession): BuildCommitMessage {
	return () => {
		const style = cfgTaskIsolationCommits.get(session.settings);
		if (style !== "ai" || !session.modelRegistry) return undefined;
		const registry = session.modelRegistry;
		const settings = session.settings;
		const sessionId = session.getSessionId?.() ?? undefined;
		return async (diff: string) => generateCommitMessage(diff, registry, settings, sessionId);
	};
}

export interface IsolatedRunOptions {
	/**
	 * Base run options handed to the subagent subprocess. This helper sets
	 * `worktree`, clears prepared/path extension preloads and custom-tool paths
	 * (isolated runs re-discover inside the worktree), and forwards everything
	 * else unchanged.
	 */
	baseOptions: ExecutorOptions;
	/** Context returned by {@link prepareIsolationContext}. */
	context: IsolationContext;
	/** PAL backend hint from `parseIsolationBackend(...)` (undefined ⇒ resolver picks). */
	preferredBackend: IsoBackendKind | undefined;
	/** Stable id used as the isolation worktree namespace and as the branch suffix. */
	agentId: string;
	/** Merge mode driving how changes are captured ("branch" commits, "patch" diffs). */
	mergeMode: "patch" | "branch";
	/** Output dir for `${agentId}.patch` artifacts (patch mode and branch-mode commit failures). */
	artifactsDir: string;
	/** Human description carried onto the branch commit (branch mode). */
	description?: string;
	/**
	 * The spawn's apply decision, persisted in the isolation sidecar so a
	 * restored run honors it (`false` ⇒ captured, never applied).
	 */
	applyChanges: boolean;
	/** Build a commit-message callback (`task.isolation.commits === "ai"`). */
	buildCommitMessage?: BuildCommitMessage;
	/**
	 * Construct a `SingleResult` when isolation setup throws — the caller has
	 * the full metadata (index, agent, assignment, modelOverride) needed to
	 * build a result shape consistent with their non-isolated path.
	 */
	buildFailureResult: (err: unknown) => SingleResult;
	/** Observe the real child result before post-run isolation work. */
	onSubprocessResult?: (result: SingleResult) => void;
}

/**
 * Write each nested-repo patch to `${artifactsDir}/${agentId}.nested-<n>-<path>.patch`
 * and return the paths. Throws on the first write failure: the caller must
 * then keep the isolation workspace alive, because it is the only other copy.
 * Every attempted destination is removed best-effort on failure — including
 * the in-progress file, which `Bun.write` may have created or truncated
 * before rejecting — so a half-written set cannot be mistaken for the
 * complete capture by the persisted-agent scanner.
 */
export async function persistNestedPatches(
	artifactsDir: string,
	agentId: string,
	nestedPatches: readonly NestedRepoPatch[],
): Promise<string[]> {
	const saved: string[] = [];
	try {
		for (const [index, nestedPatch] of nestedPatches.entries()) {
			const destination = path.join(
				artifactsDir,
				`${agentId}.nested-${index}-${nestedPatch.relativePath.replace(/[^a-zA-Z0-9._-]/g, "_") || "root"}.patch`,
			);
			// Track before writing: a mid-write failure (ENOSPC, quota) can
			// leave a truncated file behind, and `force: true` makes removing
			// a never-created path a no-op.
			saved.push(destination);
			await Bun.write(destination, nestedPatch.patch);
		}
	} catch (error) {
		await Promise.all(saved.map(file => fs.rm(file, { force: true }).catch(() => undefined)));
		throw error;
	}
	return saved;
}

interface IsolationPatchArtifacts {
	patchPath: string;
	hasRootChanges: boolean;
	nestedPatches: NestedRepoPatch[];
	nestedPatchPaths: string[];
}

interface IsolationPatchCapture {
	artifacts: IsolationPatchArtifacts;
	/** {@link deltaFingerprint} of the written delta. */
	fingerprint: bigint | number;
}

/**
 * Capture the isolation delta and write every part of it to disk — the root
 * patch and one file per nested repo — before the caller tears the workspace
 * down. Throws when any write fails so nothing captured is ever the only copy.
 */
async function writeIsolationPatch(
	isolationDir: string,
	baseline: WorktreeBaseline,
	artifactsDir: string,
	agentId: string,
): Promise<IsolationPatchCapture> {
	const delta = await captureDeltaPatch(isolationDir, baseline);
	const patchPath = path.join(artifactsDir, `${agentId}.patch`);
	await Bun.write(patchPath, delta.rootPatch);
	const nestedPatchPaths = await persistNestedPatches(artifactsDir, agentId, delta.nestedPatches);
	return {
		artifacts: {
			patchPath,
			hasRootChanges: delta.rootPatch.trim().length > 0,
			nestedPatches: delta.nestedPatches,
			nestedPatchPaths,
		},
		fingerprint: deltaFingerprint(delta.rootPatch, delta.nestedPatches),
	};
}

/**
 * Move a retained isolation workspace out of its deterministic
 * (`repoRoot` + agent id) slot into a globally unique sibling, so a later
 * isolated run with the same id cannot wipe it: `ensureIsolation`
 * unconditionally removes the deterministic base dir before writing its
 * owner marker. The owner marker, `m` mount, and backend sidecar move along,
 * so `omp worktree clear` still classifies and reclaims the workspace with
 * native teardown. Backends needing it (mounts, Btrfs subvolumes) record the
 * sidecar BEFORE the move so it travels atomically — a crash between rename
 * and a later write would leave a mounted workspace with a dead owner and
 * no metadata, and cleanup would traverse the live mount.
 */
export interface RetainedWorkspace {
	/** Workspace path to report (unique sibling on success, original dir when the move fails). */
	dir: string;
	/**
	 * False when cleanup metadata is missing that `clear` would need: the
	 * sidecar could not be written (plausible under the same disk pressure
	 * that forced retention). The error must then say the mount needs a
	 * manual unmount instead of advertising plain `worktree clear`.
	 */
	sidecarOk: boolean;
}

export async function retainIsolationWorkspace(
	isolationDir: string,
	backend?: natives.IsoBackendKind,
): Promise<RetainedWorkspace> {
	const baseDir = path.dirname(isolationDir);
	const retainedBase = `${baseDir}.retained-${Date.now().toString(36)}-${Math.floor(Math.random() * 2 ** 32).toString(16)}`;
	const needsSidecar = backend !== undefined && needsNativeTeardown(backend);
	let sidecarOk = !needsSidecar;
	if (needsSidecar && backend !== undefined) {
		try {
			await writeRetainedBackend(baseDir, backend);
			sidecarOk = true;
		} catch {
			sidecarOk = false;
		}
	}
	// A valid move can still fail transiently (Windows AV/indexer locks);
	// retry briefly before conceding the deterministic slot.
	for (let attempt = 0; attempt < 3; attempt++) {
		try {
			await fs.rename(baseDir, retainedBase);
			return { dir: path.join(retainedBase, path.basename(isolationDir)), sidecarOk };
		} catch {
			if (attempt === 2) return { dir: isolationDir, sidecarOk };
			await Bun.sleep(25);
		}
	}
	return { dir: isolationDir, sidecarOk };
}
/** Context for `isolation-error.md`: the `result.error` text for a run whose changes could not be captured or landed. */
interface IsolationErrorContext {
	kind: "merge-failed" | "patch-capture-failed" | "nested-capture-failed";
	message: string;
	captureError?: string;
	rescueBranch?: string;
	/** Set when the workspace was kept because its changes could not be written out. */
	retainedDir?: string;
	/**
	 * Set when the retained mount's unmount metadata is missing: cleanup
	 * cannot unmount before removal, so the message must direct a manual
	 * unmount instead of advertising plain `worktree clear`.
	 */
	sidecarMissing?: boolean;
}

function renderIsolationError(context: IsolationErrorContext): string {
	return prompt.render(isolationErrorTemplate, { ...context });
}

/** What {@link captureIsolatedResult} decided about a finished isolated run. */
interface IsolatedCapture {
	/** The artifact-bearing result; `undefined` when the run did not succeed, which captures nothing. */
	result: SingleResult | undefined;
	/** The workspace was kept (moved to a `.retained-*` sibling) because capture failed. */
	retained: boolean;
	/** Fingerprint of the delta handed to the caller, when one was. */
	handedOff: bigint | number | undefined;
}

interface CaptureIsolatedResultOptions {
	result: SingleResult;
	isolationDir: string;
	isolationBackend: IsoBackendKind;
	baseline: WorktreeBaseline;
	repoRoot: string;
	mergeMode: "patch" | "branch";
	artifactsDir: string;
	agentId: string;
	description: string | undefined;
	buildCommitMessage: BuildCommitMessage | undefined;
}

/**
 * Post-run tail of an isolated run, shared by live runs and runs re-adopted
 * after a restart: capture the finished sandbox's changes (branch commit or
 * patch files) onto `result`.
 */
async function captureIsolatedResult(opts: CaptureIsolatedResultOptions): Promise<IsolatedCapture> {
	const { result, isolationDir, isolationBackend, baseline, repoRoot, mergeMode, artifactsDir, agentId, description } =
		opts;
	const { buildCommitMessage } = opts;
	let workspaceRetained = false;
	let handedOff: bigint | number | undefined;
	const done = (captured: SingleResult): IsolatedCapture => ({
		result: captured,
		retained: workspaceRetained,
		handedOff,
	});
	if (mergeMode === "branch" && result.exitCode === 0) {
		let commitResult: CommitToBranchResult | null;
		try {
			commitResult = await commitToBranch(isolationDir, baseline, agentId, description, buildCommitMessage?.());
		} catch (mergeErr) {
			// Agent succeeded but the branch commit failed. `commitToBranch`
			// is not atomic: the clean-baseline path fetches the agent's
			// commits into the parent ODB and creates `omp/task/<id>` before
			// it commits the leftover working-tree delta, so a throw from
			// that trailing step leaves behind a branch that already holds
			// every commit the agent made. The isolation worktree — the only
			// other copy of those objects — is destroyed by the `finally`
			// below, so deleting the branch unconditionally turned a
			// recoverable merge conflict into permanent loss of committed
			// work (#8868). Delete only when nothing is at stake.
			const baseSha = baseline.root.headCommit;
			const branchName = `omp/task/${agentId}`;
			const rescueBranch = await rescueTaskBranch(repoRoot, branchName, baseSha);
			const msg = mergeErr instanceof Error ? mergeErr.message : String(mergeErr);
			try {
				const capture = await writeIsolationPatch(isolationDir, baseline, artifactsDir, agentId);
				return done({
					...result,
					...capture.artifacts,
					error: renderIsolationError({ kind: "merge-failed", message: msg, rescueBranch }),
				});
			} catch (patchErr) {
				workspaceRetained = true;
				const retained = await retainIsolationWorkspace(isolationDir, isolationBackend);
				return done({
					...result,
					error: renderIsolationError({
						kind: "merge-failed",
						message: msg,
						captureError: patchErr instanceof Error ? patchErr.message : String(patchErr),
						rescueBranch,
						retainedDir: retained.dir,
						sidecarMissing: !retained.sidecarOk,
					}),
				});
			}
		}
		handedOff = deltaFingerprint(commitResult?.rootPatch ?? "", commitResult?.nestedPatches ?? []);
		// The branch holds the root-repo work, but nested-repo patches exist
		// only in memory until written; the workspace goes away in `finally`.
		try {
			const nestedPatchPaths = await persistNestedPatches(artifactsDir, agentId, commitResult?.nestedPatches ?? []);
			return done({
				...result,
				branchName: commitResult?.branchName,
				branchBaseSha: commitResult?.baseSha,
				nestedPatches: commitResult?.nestedPatches,
				nestedPatchPaths,
			});
		} catch (persistErr) {
			workspaceRetained = true;
			const retained = await retainIsolationWorkspace(isolationDir, isolationBackend);
			return done({
				...result,
				branchName: commitResult?.branchName,
				branchBaseSha: commitResult?.baseSha,
				nestedPatches: commitResult?.nestedPatches,
				error: renderIsolationError({
					kind: "nested-capture-failed",
					message: persistErr instanceof Error ? persistErr.message : String(persistErr),
					retainedDir: retained.dir,
					sidecarMissing: !retained.sidecarOk,
				}),
			});
		}
	}
	if (result.exitCode === 0) {
		try {
			const capture = await writeIsolationPatch(isolationDir, baseline, artifactsDir, agentId);
			handedOff = capture.fingerprint;
			return done({ ...result, ...capture.artifacts });
		} catch (patchErr) {
			workspaceRetained = true;
			const retained = await retainIsolationWorkspace(isolationDir, isolationBackend);
			return done({
				...result,
				error: renderIsolationError({
					kind: "patch-capture-failed",
					message: patchErr instanceof Error ? patchErr.message : String(patchErr),
					retainedDir: retained.dir,
					sidecarMissing: !retained.sidecarOk,
				}),
			});
		}
	}
	return { result: undefined, retained: workspaceRetained, handedOff };
}

/**
 * Run a subagent inside an isolation worktree and capture its changes.
 *
 * Branch mode: on success, commits the diff onto `omp/task/${agentId}` and
 * returns `branchName` + `nestedPatches` (+ `nestedPatchPaths`). On commit
 * failure the still-live isolation diff is written to
 * `${artifactsDir}/${agentId}.patch`, the task branch is kept when it already
 * carries commits (deleted otherwise), and `result.error` carries the
 * merge-failure message plus recovery hint.
 *
 * Patch mode: on success, writes `${artifactsDir}/${agentId}.patch` plus one
 * `${agentId}.nested-<n>-<path>.patch` per nested repo and returns
 * `patchPath` + `nestedPatches` + `nestedPatchPaths`.
 *
 * Failure paths preserve the underlying `SingleResult` whenever possible so
 * the caller can still surface the subagent's output; only isolation setup
 * itself routes through {@link IsolatedRunOptions.buildFailureResult}.
 *
 * Kept-alive runs retain the isolation handle through idle/parked lifecycle
 * transitions, then capture final changes and clean up on release. One-shot
 * and failed startup paths clean up in `finally`. If captured changes cannot
 * be written to disk, the workspace is retained under a unique `.retained-*`
 * sibling and its path is named in the resulting error.
 */
export async function runIsolatedSubprocess(opts: IsolatedRunOptions): Promise<SingleResult> {
	let taskBaseline: WorktreeBaseline | undefined;
	/** Fingerprint of the delta already handed to the caller at run end; release only re-captures on change. */
	let handedOff: bigint | number | undefined;
	let handle: IsolationHandle | undefined;
	let deferredCleanup: Promise<void> | undefined;
	let retainWorkspace = false;
	let baseReleasePromise: Promise<void> | undefined;
	let cleanupPromise: Promise<void> | undefined;
	let releasePromise: Promise<void> | undefined;
	/** The run's IRC wake hold, released only once the run-end capture is done. */
	let releaseIrcWakes: (() => void) | undefined;
	const releaseBase = (): Promise<void> => {
		baseReleasePromise ??= opts.baseOptions.onRelease?.() ?? Promise.resolve();
		return baseReleasePromise;
	};
	const cleanupHandle = (): Promise<void> => {
		cleanupPromise ??= (async () => {
			await releaseBase();
			if (handle && !retainWorkspace) {
				await cleanupIsolation(handle);
				await removeIsolationSidecar(opts.artifactsDir, opts.agentId);
			}
		})();
		return cleanupPromise;
	};
	const releaseIsolation = (): Promise<void> => {
		releasePromise ??= (async () => {
			if (!handle || !taskBaseline || retainWorkspace) {
				await cleanupHandle();
				return;
			}
			try {
				let capture: IsolationPatchCapture;
				try {
					capture = await writeIsolationPatch(handle.mergedDir, taskBaseline, opts.artifactsDir, opts.agentId);
				} catch (captureErr) {
					retainWorkspace = true;
					const retained = await retainIsolationWorkspace(handle.mergedDir, handle.backend);
					throw new Error(
						renderIsolationError({
							kind: "patch-capture-failed",
							message: captureErr instanceof Error ? captureErr.message : String(captureErr),
							retainedDir: retained.dir,
							sidecarMissing: !retained.sidecarOk,
						}),
					);
				}
				const patchResult = capture.artifacts;
				AgentRegistry.global().setHistory(opts.agentId, {
					patchPath: patchResult.patchPath,
					nestedPatchPaths: patchResult.nestedPatchPaths,
				});
				// Nothing changed since the run-end capture the caller already
				// merged or applied: a branch now would only duplicate that work.
				if (capture.fingerprint === handedOff) return;
				const commitResult = await commitToBranch(
					handle.mergedDir,
					taskBaseline,
					opts.agentId,
					opts.description,
					undefined,
				);
				AgentRegistry.global().setHistory(opts.agentId, {
					patchPath: patchResult.patchPath,
					branchName: commitResult?.branchName,
					nestedPatchPaths: patchResult.nestedPatchPaths,
				});
			} finally {
				await cleanupHandle();
			}
		})();
		return releasePromise;
	};
	try {
		handle = await ensureIsolation(opts.context.repoRoot, opts.agentId, opts.preferredBackend);
		const isolationDir = handle.mergedDir;
		const isolationBackend = handle.backend;
		const baseline = await captureIsolationBaseline(isolationDir, opts.context.repoRoot);
		taskBaseline = baseline;
		// Persist the setup facts BEFORE the subprocess starts: a crash mid-run
		// must leave enough on disk for `--restore` to re-adopt the sandbox. A
		// write failure fails isolation setup (catch below; finally cleans up).
		await writeIsolationSidecar(opts.artifactsDir, {
			version: 1,
			agentId: opts.agentId,
			backend: isolationBackend,
			workspacePath: isolationDir,
			repoRoot: opts.context.repoRoot,
			baseline,
			mergeMode: opts.mergeMode,
			applyChanges: opts.applyChanges,
			...(opts.description !== undefined ? { description: opts.description } : {}),
		});
		const result = await runSubprocess({
			...opts.baseOptions,
			worktree: isolationDir,
			preloadedExtensionPaths: undefined,
			preloadedPreparedExtensions: undefined,
			preloadedCustomToolPaths: undefined,
			onCleanupDeferred: completion => {
				deferredCleanup = completion;
				opts.baseOptions.onCleanupDeferred?.(completion);
			},
			// One-shot runs get `releaseBase` (never touches the worktree): their
			// `finalizeSubagentLifecycle` calls `onRelease` before this function's
			// post-run capture, so the handle must survive until the `finally`
			// below cleans it up. Only kept-alive runs hand full capture+cleanup
			// (`releaseIsolation`) to the agent lifecycle.
			onRelease: opts.baseOptions.keepAlive === false ? releaseBase : releaseIsolation,
			deferIrcWakeRelease: release => {
				releaseIrcWakes = release;
			},
		});
		opts.onSubprocessResult?.(result);
		// A successful result cannot be captured while deferred owner jobs or
		// shutdown hooks may still write the worktree. Failed runs skip capture,
		// so their cleanup remains asynchronous.
		if (deferredCleanup && result.exitCode === 0) {
			await deferredCleanup;
		}
		const captured = await captureIsolatedResult({
			result,
			isolationDir,
			isolationBackend,
			baseline,
			repoRoot: opts.context.repoRoot,
			mergeMode: opts.mergeMode,
			artifactsDir: opts.artifactsDir,
			agentId: opts.agentId,
			description: opts.description,
			buildCommitMessage: opts.buildCommitMessage,
		});
		if (captured.retained) retainWorkspace = true;
		if (captured.handedOff !== undefined) handedOff = captured.handedOff;
		if (captured.result) return rememberAgentArtifacts(captured.result);
		return rememberAgentArtifacts(result);
	} catch (err) {
		return rememberAgentArtifacts(opts.buildFailureResult(err));
	} finally {
		if (
			handle &&
			!retainWorkspace &&
			!releasePromise &&
			!(opts.baseOptions.keepAlive !== false && AgentLifecycleManager.global().has(opts.agentId))
		) {
			if (deferredCleanup) {
				trackLateCleanup(deferredCleanup.then(cleanupHandle), {
					agentId: opts.agentId,
					resource: "isolation",
				});
			} else {
				await cleanupHandle();
			}
		}
		// The worktree is captured: a held peer message may now wake a turn in it.
		if (releaseIrcWakes && opts.baseOptions.deferIrcWakeRelease)
			opts.baseOptions.deferIrcWakeRelease(releaseIrcWakes);
		else releaseIrcWakes?.();
	}
}

export interface IsolationMergeOptions {
	result: SingleResult;
	repoRoot: string;
	mergeMode: "patch" | "branch";
}

export interface IsolationMergeOutcome {
	/** Trailing summary appended to the subagent's result text. May be empty. */
	summary: string;
	/**
	 * Tri-state apply outcome:
	 * - `true`  — merge ran (or had nothing to apply) and left the repo clean.
	 * - `false` — merge attempted and failed; artifacts are preserved.
	 * - `null`  — caller skipped the merge phase entirely (e.g. `apply=false`).
	 */
	changesApplied: boolean | null;
	hadAnyChanges: boolean;
	/** True iff the root branch actually merged — gates nested-repo patch application. */
	mergedBranchForNestedPatches: boolean;
}

/**
 * Apply changes captured by {@link runIsolatedSubprocess} back to the parent
 * repo: patch apply (patch mode) or cherry-pick + cleanup (branch mode).
 *
 * The caller decides whether to run this at all — eval `agent()` with
 * `apply=False` skips this step and surfaces the patch artifact / branch name
 * instead.
 */
export async function mergeIsolatedChanges(opts: IsolationMergeOptions): Promise<IsolationMergeOutcome> {
	const { result, repoRoot, mergeMode } = opts;
	const repo = vcs.requireGit(repoRoot);
	try {
		if (mergeMode === "branch") {
			if (!result.branchName && result.exitCode === 0 && !result.aborted && result.error) {
				return {
					summary: renderIsolationSummary({
						kind: "branch-capture-failed",
						error: result.error,
						rootPatchPath: result.patchPath,
						nestedPatchPaths: result.nestedPatchPaths,
					}),
					changesApplied: false,
					hadAnyChanges: false,
					mergedBranchForNestedPatches: false,
				};
			}
			const canApplyNestedOnly =
				!result.branchName && result.exitCode === 0 && !result.aborted && (result.nestedPatches?.length ?? 0) > 0;
			if (!result.branchName || result.exitCode !== 0 || result.aborted) {
				return {
					summary: canApplyNestedOnly
						? "\n\nNo root changes to apply; nested repository patches captured."
						: "\n\nNo changes to apply.",
					changesApplied: true,
					hadAnyChanges: canApplyNestedOnly,
					mergedBranchForNestedPatches: canApplyNestedOnly,
				};
			}
			const mergeResult = await mergeTaskBranches(repoRoot, [
				{
					branchName: result.branchName,
					taskId: result.id,
					description: result.description,
					baseSha: result.branchBaseSha,
				},
			]);
			const mergedBranchForNestedPatches = mergeResult.merged.includes(result.branchName);
			const changesApplied = mergeResult.failed.length === 0;
			const hadAnyChanges = changesApplied && mergeResult.merged.length > 0;

			let summary: string;
			if (changesApplied) {
				summary = hadAnyChanges ? `\n\nMerged branch: ${result.branchName}` : "\n\nNo changes to apply.";
			} else {
				// The nested patches are skipped when the branch did not merge; name
				// their files so the parent can recover them alongside the branch.
				summary = renderIsolationSummary({
					kind: "branch-merge-failed",
					branchName: result.branchName,
					conflict: mergeResult.conflict,
					nestedPatchPaths: result.nestedPatchPaths,
				});
			}

			// Clean up the merged branch (keep failed ones for manual resolution)
			if (changesApplied) {
				await cleanupTaskBranches(repoRoot, [result.branchName]);
			}
			return { summary, changesApplied, hadAnyChanges, mergedBranchForNestedPatches };
		}

		// Patch mode: apply the patch from a successful run. A failed or
		// aborted run has nothing to apply and must not block the result.
		let changesApplied: boolean;
		let hadAnyChanges: boolean;
		let conflictPaths: string[] | undefined;
		const succeeded = result.exitCode === 0 && !result.error && !result.aborted;
		if (!succeeded) {
			changesApplied = true;
			hadAnyChanges = false;
		} else if (!result.patchPath) {
			changesApplied = false;
			hadAnyChanges = false;
		} else {
			const patchText = await Bun.file(result.patchPath).text();
			if (!patchText.trim()) {
				changesApplied = true;
				hadAnyChanges = false;
			} else {
				const normalized = patchText.endsWith("\n") ? patchText : `${patchText}\n`;
				// Serialize check+apply per repo: native apply is read-map -> apply ->
				// write-map, so concurrent sibling merge-backs would lose updates.
				// Branch mode locks inside `mergeTaskBranches`; never nest here.
				const outcome = await withRepoLock(repoRoot, async () => {
					// Idempotence: declare a no-op only when the reverse patch applies AND
					// the forward patch does not. `--reverse --check` alone can theoretically
					// succeed if the file happens to carry the postimage at another location
					// via git-apply's fuzz factor; requiring the forward check to fail
					// removes that ambiguity while still catching true already-applied
					// runs. Reads only — neither call touches the worktree.
					const [alreadyApplied, forwardApplies] = await Promise.all([
						probePatchApplies(repo, normalized, { reverse: true }),
						probePatchApplies(repo, normalized, {}),
					]);
					if (alreadyApplied && !forwardApplies) return "already-applied" as const;
					return applyPatchWithThreeWayFallback(repo, normalized, forwardApplies);
				});
				hadAnyChanges = outcome === "applied";
				changesApplied = outcome !== "conflict";
				if (outcome === "conflict") conflictPaths = patchTouchedFiles(normalized);
			}
		}

		let summary: string;
		if (changesApplied) {
			summary = hadAnyChanges ? "\n\nApplied patches: yes" : "\n\nNo changes to apply.";
		} else if (conflictPaths) {
			// Tree is untouched (native apply is all-or-nothing); the patch artifact stays.
			summary = renderIsolationSummary({
				kind: "patch-conflict",
				conflictPaths,
				rootPatchPath: result.patchPath,
				nestedPatchPaths: result.nestedPatchPaths,
			});
		} else {
			// Nested apply is skipped when the root patch did not apply; the
			// persisted nested patches are the parent's only pointer to that work.
			summary = renderIsolationSummary({
				kind: "not-applied",
				rootPatchPath: result.patchPath,
				nestedPatchPaths: result.nestedPatchPaths,
			});
		}
		return { summary, changesApplied, hadAnyChanges, mergedBranchForNestedPatches: false };
	} catch (mergeErr) {
		return {
			summary: renderIsolationSummary({
				kind: "merge-error",
				error: mergeErr instanceof Error ? mergeErr.message : String(mergeErr),
				branchName: result.branchName,
				rootPatchPath: result.patchPath,
				nestedPatchPaths: result.nestedPatchPaths,
			}),
			changesApplied: false,
			hadAnyChanges: false,
			mergedBranchForNestedPatches: false,
		};
	}
}

export interface NestedPatchApplyOptions {
	/** Subagent result carrying `nestedPatches`/`exitCode`/`aborted`. */
	result: SingleResult;
	repoRoot: string;
	mergeMode: "patch" | "branch";
	/** Parent merge outcome — patch mode skips nested apply when this is `false`. */
	changesApplied: boolean | null;
	/** Branch mode gates nested apply on whether the root branch merged. */
	mergedBranchForNestedPatches: boolean;
	/** Optional AI commit-message callback for nested commits; falls back to a generic message. */
	commitMessage?: (diff: string) => Promise<string | null>;
}

/**
 * Apply nested-repo patches after the parent merge phase. Centralizes the
 * three-way gate (exitCode/aborted, patch-mode failed parent, branch-mode
 * branch-merged) and the non-fatal failure handling so `TaskTool` and the
 * eval `agent()` bridge use one implementation.
 *
 * Returns a system-notification suffix to append to the parent merge summary,
 * or an empty string when nothing was applied or the nested apply succeeded.
 */
export async function applyEligibleNestedPatches(opts: NestedPatchApplyOptions): Promise<string> {
	const { result, repoRoot, mergeMode, changesApplied, mergedBranchForNestedPatches, commitMessage } = opts;
	if (mergeMode === "patch" && changesApplied === false) return "";
	const nestedPatches = result.nestedPatches ?? [];
	const eligible =
		nestedPatches.length > 0 &&
		result.exitCode === 0 &&
		!result.aborted &&
		(mergeMode !== "branch" || mergedBranchForNestedPatches);
	if (!eligible) return "";
	try {
		const warnings = await applyNestedPatches(repoRoot, nestedPatches, commitMessage);
		if (warnings.length === 0) return "";
		return `\n\n<system-notification>${warnings.join("\n")}</system-notification>`;
	} catch (applyErr) {
		// Nested patch failures are non-fatal to the parent merge, but the patch
		// files are the only surviving copy of that work — name them.
		return renderIsolationSummary({
			kind: "nested-apply-failed",
			error: applyErr instanceof Error ? applyErr.message : String(applyErr),
			nestedPatchPaths: result.nestedPatchPaths,
		});
	}
}

/**
 * Render the `apply=false` outcome: where the captured changes were left,
 * with no merge attempted.
 */
export function describeCapturedChanges(result: SingleResult): string {
	const nestedPatchPaths = result.nestedPatchPaths ?? [];
	return renderIsolationSummary({
		kind: "captured",
		branchName: result.branchName,
		rootPatchPath: result.hasRootChanges === false ? undefined : result.patchPath,
		nestedCount: nestedPatchPaths.length || (result.nestedPatches?.length ?? 0),
		nestedPatchPaths,
	});
}

export interface FinishAdoptedIsolationOptions {
	sidecar: IsolationSidecar;
	handle: IsolationHandle;
	/** The restored turn's result; must be a successful one. */
	result: SingleResult;
	artifactsDir: string;
}

/**
 * Finish a sandbox re-adopted by `adoptIsolatedSandbox` after its interrupted
 * turn completed: the same capture tail a live run uses, then (only when the
 * original spawn had `apply` on) the exported merge, with the summaries
 * appended to `result.output`. An `apply=false` spawn is captured but never
 * merged. The sandbox is cleaned up unless capture failed and retained it.
 * Callers pass only successful results; a failed turn leaves the sandbox alone.
 */
export async function finishAdoptedIsolation(opts: FinishAdoptedIsolationOptions): Promise<SingleResult> {
	const { sidecar, handle, artifactsDir } = opts;
	const captured = await captureIsolatedResult({
		result: opts.result,
		isolationDir: handle.mergedDir,
		isolationBackend: handle.backend,
		baseline: sidecar.baseline,
		repoRoot: sidecar.repoRoot,
		mergeMode: sidecar.mergeMode,
		artifactsDir,
		agentId: sidecar.agentId,
		description: sidecar.description,
		buildCommitMessage: undefined,
	});
	if (!captured.result) throw new Error(`Adopted isolation for ${sidecar.agentId} was given a non-successful result`);
	let result = rememberAgentArtifacts(captured.result);
	let summary = "";
	if (result.error) {
		// Capture failed: the changes were not landed; `error` names the retained workspace.
		summary = renderIsolationSummary({
			kind: "capture-error",
			error: result.error,
			branchName: result.branchName,
			rootPatchPath: result.hasRootChanges === false ? undefined : result.patchPath,
			nestedPatchPaths: result.nestedPatchPaths ?? [],
		});
	} else if (!sidecar.applyChanges) {
		summary = describeCapturedChanges(result);
	} else {
		const outcome = await mergeIsolatedChanges({
			result,
			repoRoot: sidecar.repoRoot,
			mergeMode: sidecar.mergeMode,
		});
		summary = outcome.summary;
		if (outcome.changesApplied !== false) {
			summary += await applyEligibleNestedPatches({
				result,
				repoRoot: sidecar.repoRoot,
				mergeMode: sidecar.mergeMode,
				changesApplied: outcome.changesApplied,
				mergedBranchForNestedPatches: outcome.mergedBranchForNestedPatches,
			});
		}
	}
	result = { ...result, output: `${result.output}${summary}` };
	if (!captured.retained) {
		await cleanupIsolation(handle);
		await removeIsolationSidecar(artifactsDir, sidecar.agentId);
	}
	return result;
}
