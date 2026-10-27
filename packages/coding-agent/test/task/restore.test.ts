import { afterEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import type { AssistantMessage, Message } from "@oh-my-pi/pi-ai";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry, MAIN_AGENT_ID } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { ensurePersistedRoster } from "@oh-my-pi/pi-coding-agent/registry/persisted-agents";
import { getAgentTombstonePath } from "@oh-my-pi/pi-coding-agent/registry/agent-tombstone";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { cfgWorkspaceAdditionalDirectories } from "@oh-my-pi/pi-coding-agent/session/context-settings";
import { MCPManager } from "@oh-my-pi/pi-coding-agent/mcp/manager";
import type { CreateAgentSessionOptions } from "@oh-my-pi/pi-coding-agent/sdk";
import * as sdkModule from "@oh-my-pi/pi-coding-agent/sdk";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import { restoreSession } from "@oh-my-pi/pi-coding-agent/task/restore";
import { createPersistedSubagentReviverFactory } from "@oh-my-pi/pi-coding-agent/task/persisted-revive";
import {
	adoptIsolatedSandbox,
	captureIsolationBaseline,
	ensureIsolation,
	forgetAdoptedSandbox,
	getAdoptedSandbox,
	type IsolationSidecar,
	readIsolationSidecar,
	writeIsolationSidecar,
} from "@oh-my-pi/pi-coding-agent/task/worktree";
import { ISOLATION_OWNER_FILE } from "@oh-my-pi/pi-coding-agent/task/isolation-ownership";
import * as natives from "@oh-my-pi/pi-natives";
import type { SingleResult } from "@oh-my-pi/pi-tui/tools/task";
import { logger, removeWithRetries, setWorktreesDir, TempDir } from "@oh-my-pi/pi-utils";

const zeroUsage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function assistant(fields: Partial<AssistantMessage>): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4-5",
		usage: zeroUsage,
		stopReason: "stop",
		timestamp: Date.now(),
		...fields,
	};
}

/** A turn the process died in, mid tool call (no result, no exit record). */
const cutOff: Message[] = [
	{ role: "user", content: "work", timestamp: Date.now() },
	assistant({
		content: [{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "a.ts" } }],
		stopReason: "toolUse",
	}),
];
/** A turn that ended on the model's own answer. */
const finished: Message[] = [
	{ role: "user", content: "work", timestamp: Date.now() },
	assistant({ content: [{ type: "text", text: "all done" }] }),
];

/** Writes subagent `id`'s transcript under the main session `mainFile`, with its spawn contract. */
async function writeSubagent(
	mainFile: string,
	cwd: string,
	id: string,
	messages: Message[],
	isolated = false,
): Promise<string> {
	const file = path.join(mainFile.slice(0, -6), `${id}.jsonl`);
	const manager = await SessionManager.open(file, undefined, undefined, {
		initialCwd: cwd,
		parentSession: mainFile,
		suppressBreadcrumb: true,
	});
	manager.appendSessionInit({
		systemPrompt: ["sub prompt"],
		task: `${id} task`,
		tools: ["read", "yield"],
		...(isolated ? { isolated: true } : {}),
	});
	for (const message of messages) manager.appendMessage(message);
	await manager.close();
	return file;
}

describe("restoreSession (omp --restore)", () => {
	const ids = ["Cut", "Done", "Killed"];
	let tempDir: TempDir | undefined;
	let manager: AsyncJobManager | undefined;

	afterEach(async () => {
		vi.restoreAllMocks();
		for (const id of ids) AgentRegistry.global().unregister(id);
		await manager?.dispose({ timeoutMs: 1000 });
		manager = undefined;
		tempDir?.removeSync();
		tempDir = undefined;
	});

	it("goes on with each subagent cut off mid-turn as a job its parent receives, not with finished or killed ones", async () => {
		tempDir = TempDir.createSync("@pi-restore-subagents-");
		const cwd = tempDir.path();
		const main = SessionManager.create(cwd, path.join(cwd, "sessions"));
		main.appendMessage({ role: "user", content: "fan out", timestamp: Date.now() });
		const mainFile = main.getSessionFile();
		if (!mainFile) throw new Error("Expected a persisted session file");
		await main.close();
		await writeSubagent(mainFile, cwd, "Cut", cutOff);
		await writeSubagent(mainFile, cwd, "Done", finished);
		const killed = await writeSubagent(mainFile, cwd, "Killed", cutOff);
		await fs.writeFile(getAgentTombstonePath(killed), "");

		manager = new AsyncJobManager({});
		const delivered: string[] = [];
		manager.registerDeliverySink(MAIN_AGENT_ID, jobId => {
			delivered.push(jobId);
		});
		const result: SingleResult = {
			index: 0,
			id: "Cut",
			agent: "Cut",
			agentSource: "user",
			task: "Cut task",
			exitCode: 0,
			output: "finished after the restart",
			stderr: "",
			truncated: false,
			durationMs: 5,
			tokens: 0,
			requests: 1,
		};
		const followUp = vi.spyOn(executorModule, "runSubagentFollowUpTurn").mockResolvedValue(result);
		const restoreTurn = vi.fn(async () => true);
		const session = {
			sessionManager: { getSessionFile: () => mainFile },
			asyncJobManager: manager,
			settings: Settings.isolated(),
			restoreInterruptedTurn: restoreTurn,
		} as unknown as AgentSession;

		const outcome = await restoreSession(session);

		expect(outcome).toEqual({ turn: true, subagents: ["Cut"], unrestorable: [] });
		expect(followUp).toHaveBeenCalledTimes(1);
		expect(followUp.mock.calls[0]![0]).toMatchObject({ id: "Cut", restore: true, message: "Cut task" });
		await manager.getJob("Cut")!.promise;
		await manager.drainDeliveries({ filter: { ownerId: MAIN_AGENT_ID } });
		expect(delivered).toEqual(["Cut"]);
	});
});

async function runGit(repo: string, args: string[]): Promise<string> {
	const proc = Bun.spawn(["git", ...args], { cwd: repo, stderr: "pipe", stdout: "pipe", windowsHide: true });
	const [stdout, stderr, exitCode] = await Promise.all([
		new Response(proc.stdout).text(),
		new Response(proc.stderr).text(),
		proc.exited,
	]);
	if (exitCode !== 0) throw new Error(stderr.trim() || `git ${args.join(" ")} failed`);
	return stdout.trim();
}

/** A pid that is certainly not running: a child that already exited. */
async function deadPid(): Promise<number> {
	const proc = Bun.spawn(["true"]);
	await proc.exited;
	return proc.pid;
}

describe("isolated subagent restore (sidecar re-adoption)", () => {
	const id = "Iso";
	let tempDir: TempDir | undefined;
	let manager: AsyncJobManager | undefined;
	const roots: string[] = [];

	afterEach(async () => {
		vi.restoreAllMocks();
		MCPManager.resetForTests();
		AgentRegistry.global().unregister(id);
		await AgentLifecycleManager.global().dispose();
		await manager?.dispose({ timeoutMs: 1000 });
		manager = undefined;
		setWorktreesDir(undefined);
		tempDir?.removeSync();
		tempDir = undefined;
		await Promise.all(roots.splice(0).map(dir => removeWithRetries(dir)));
	});

	/**
	 * A parent repo, a real sandbox of it with an uncommitted agent edit, its
	 * sidecar and a crashed owner's marker, plus the isolated subagent transcript.
	 */
	async function crashedIsolatedRun(applyChanges: boolean) {
		const root = await fs.mkdtemp(path.join(os.tmpdir(), "omp-restore-iso-"));
		roots.push(root);
		const repo = path.join(root, "repo");
		await fs.mkdir(repo);
		await runGit(repo, ["init", "-q", "-b", "main"]);
		await runGit(repo, ["config", "core.autocrlf", "false"]);
		await runGit(repo, ["config", "user.email", "t@example.com"]);
		await runGit(repo, ["config", "user.name", "t"]);
		await Bun.write(path.join(repo, "a.txt"), "one\ntwo\nthree\n");
		await runGit(repo, ["add", "."]);
		await runGit(repo, ["commit", "-q", "-m", "initial"]);

		delete process.env.OMP_WORKTREE_DIR;
		setWorktreesDir(path.join(root, "wt"));
		vi.spyOn(natives, "isoResolve").mockResolvedValue({
			kind: natives.IsoBackendKind.Rcopy,
			candidates: [natives.IsoBackendKind.Rcopy],
			fellBack: false,
			reason: undefined,
		});
		const handle = await ensureIsolation(repo, id, natives.IsoBackendKind.Rcopy);
		const baseline = await captureIsolationBaseline(handle.mergedDir, repo);
		await Bun.write(path.join(handle.mergedDir, "a.txt"), "one\nTWO\nthree\n");

		const cwd = path.join(root, "session");
		await fs.mkdir(cwd);
		const main = SessionManager.create(cwd, path.join(cwd, "sessions"));
		main.appendMessage({ role: "user", content: "fan out", timestamp: Date.now() });
		const mainFile = main.getSessionFile();
		if (!mainFile) throw new Error("Expected a persisted session file");
		await main.close();
		const subFile = await writeSubagent(mainFile, handle.mergedDir, id, cutOff, true);
		const artifactsDir = path.dirname(subFile);
		const sidecar: IsolationSidecar = {
			version: 1,
			agentId: id,
			backend: handle.backend,
			workspacePath: handle.mergedDir,
			repoRoot: repo,
			baseline,
			mergeMode: "patch",
			applyChanges,
		};
		await writeIsolationSidecar(artifactsDir, sidecar);
		const ownerFile = path.join(path.dirname(handle.mergedDir), ISOLATION_OWNER_FILE);
		await Bun.write(ownerFile, JSON.stringify({ pid: await deadPid(), id }));
		return { repo, handle, mainFile, artifactsDir, sidecar, ownerFile };
	}

	function restoringSession(mainFile: string): AgentSession {
		manager = new AsyncJobManager({});
		manager.registerDeliverySink(MAIN_AGENT_ID, () => {});
		return {
			sessionManager: { getSessionFile: () => mainFile },
			asyncJobManager: manager,
			settings: Settings.isolated(),
			restoreInterruptedTurn: async () => false,
		} as unknown as AgentSession;
	}

	const turnResult: SingleResult = {
		index: 0,
		id,
		agent: id,
		agentSource: "user",
		task: `${id} task`,
		exitCode: 0,
		output: "finished after the restart",
		stderr: "",
		truncated: false,
		durationMs: 5,
		tokens: 0,
		requests: 1,
	};

	it("round-trips the sidecar with nested baselines and rejects a malformed one", async () => {
		const dir = await fs.mkdtemp(path.join(os.tmpdir(), "omp-sidecar-"));
		roots.push(dir);
		const repoBaseline = (repoRoot: string) => ({
			repoRoot,
			headCommit: "abc123",
			staged: "",
			unstaged: "",
			untracked: ["new.txt"],
			untrackedPatch: "",
		});
		const sidecar: IsolationSidecar = {
			version: 1,
			agentId: "Rt",
			backend: natives.IsoBackendKind.Rcopy,
			workspacePath: "/wt/t1/m",
			repoRoot: "/repo",
			baseline: {
				root: repoBaseline("/repo"),
				nested: [{ relativePath: "pkg/inner", baseline: repoBaseline("/repo/pkg/inner") }],
			},
			mergeMode: "branch",
			applyChanges: false,
			description: "label",
		};
		expect(await readIsolationSidecar(dir, "Rt")).toBeUndefined();
		await writeIsolationSidecar(dir, sidecar);
		expect(await readIsolationSidecar(dir, "Rt")).toEqual(sidecar);
		await Bun.write(path.join(dir, "Bad.isolation.json"), "{ not json");
		await expect(readIsolationSidecar(dir, "Bad")).rejects.toThrow();
		await Bun.write(path.join(dir, "Old.isolation.json"), JSON.stringify({ version: 2 }));
		await expect(readIsolationSidecar(dir, "Old")).rejects.toThrow(/unsupported version/);
		const { baseline: _omitted, ...truncated } = sidecar;
		await Bun.write(path.join(dir, "Cut.isolation.json"), JSON.stringify(truncated));
		await expect(readIsolationSidecar(dir, "Cut")).rejects.toThrow(/baseline missing/);

		// Each field restore relies on is validated, so a corrupt sidecar is refused
		// at read time instead of driving adoption/merge with a wrong value.
		const rows: Array<[string, Record<string, unknown>, RegExp]> = [
			["EmptyId", { agentId: "" }, /agentId must be a non-empty string/],
			["NoWorkspace", { workspacePath: 42 }, /workspacePath must be a non-empty string/],
			["BadBackend", { backend: "rcopy" }, /backend must be a number/],
			["BadMode", { mergeMode: "squash" }, /mergeMode must be patch or branch/],
			["BadApply", { applyChanges: "yes" }, /applyChanges must be a boolean/],
			["BadDesc", { description: 7 }, /description must be a string/],
			[
				"NoHead",
				{ baseline: { root: { ...sidecar.baseline.root, headCommit: "" }, nested: [] } },
				/baseline\.root\.headCommit must be a non-empty string/,
			],
			["NoNested", { baseline: { root: sidecar.baseline.root } }, /baseline\.nested must be an array/],
		];
		for (const [name, patch, message] of rows) {
			await Bun.write(path.join(dir, `${name}.isolation.json`), JSON.stringify({ ...sidecar, ...patch }));
			await expect(readIsolationSidecar(dir, name)).rejects.toThrow(message);
		}
	});

	it("re-adopts the sandbox, captures the agent's edit and lands it in the repo when the spawn applied changes", async () => {
		const { repo, mainFile, artifactsDir, handle } = await crashedIsolatedRun(true);
		const followUp = vi.spyOn(executorModule, "runSubagentFollowUpTurn").mockResolvedValue(turnResult);

		const outcome = await restoreSession(restoringSession(mainFile));

		expect(outcome.subagents).toEqual([id]);
		expect(outcome.unrestorable).toEqual([]);
		expect(followUp).toHaveBeenCalledTimes(1);
		await manager!.getJob(id)!.promise;
		expect(await Bun.file(path.join(artifactsDir, `${id}.patch`)).text()).toContain("+TWO");
		expect(await Bun.file(path.join(repo, "a.txt")).text()).toBe("one\nTWO\nthree\n");
		await expect(fs.stat(handle.mergedDir)).rejects.toThrow();
		expect(await readIsolationSidecar(artifactsDir, id)).toBeUndefined();
	});

	it("captures but never applies on restore when the spawn was apply=false", async () => {
		const { repo, mainFile, artifactsDir, handle } = await crashedIsolatedRun(false);
		vi.spyOn(executorModule, "runSubagentFollowUpTurn").mockResolvedValue(turnResult);

		const outcome = await restoreSession(restoringSession(mainFile));

		expect(outcome.subagents).toEqual([id]);
		const job = manager!.getJob(id)!;
		await job.promise;
		expect(await Bun.file(path.join(artifactsDir, `${id}.patch`)).text()).toContain("+TWO");
		expect(await Bun.file(path.join(repo, "a.txt")).text()).toBe("one\ntwo\nthree\n");
		expect(job.resultText).toContain("apply=false");
		await expect(fs.stat(handle.mergedDir)).rejects.toThrow();
	});

	for (const [label, failure] of [
		["failed", { exitCode: 1, error: "model gave up" }],
		["aborted", { aborted: true }],
	] as const) {
		it(`keeps the sandbox untouched and names it when the restored turn ${label}`, async () => {
			const { repo, mainFile, artifactsDir, handle } = await crashedIsolatedRun(true);
			vi.spyOn(executorModule, "runSubagentFollowUpTurn").mockResolvedValue({ ...turnResult, ...failure });

			const outcome = await restoreSession(restoringSession(mainFile));

			expect(outcome.subagents).toEqual([id]);
			const job = manager!.getJob(id)!;
			await job.promise;
			expect(job.status).toBe("failed");
			expect(job.errorText).toContain(`Isolated workspace kept at ${handle.mergedDir}`);
			// Nothing captured, nothing applied, the agent's edit still sits in the sandbox.
			expect(await Bun.file(path.join(artifactsDir, `${id}.patch`)).exists()).toBe(false);
			expect(await Bun.file(path.join(repo, "a.txt")).text()).toBe("one\ntwo\nthree\n");
			expect(await Bun.file(path.join(handle.mergedDir, "a.txt")).text()).toBe("one\nTWO\nthree\n");
		});
	}

	it("reports a deleted sandbox as unrestorable with the reason and does not run the turn", async () => {
		const { mainFile, handle } = await crashedIsolatedRun(true);
		await fs.rm(path.dirname(handle.mergedDir), { recursive: true, force: true });
		const followUp = vi.spyOn(executorModule, "runSubagentFollowUpTurn").mockResolvedValue(turnResult);

		const outcome = await restoreSession(restoringSession(mainFile));

		expect(outcome.subagents).toEqual([]);
		expect(outcome.unrestorable).toHaveLength(1);
		expect(outcome.unrestorable[0]).toMatchObject({ id });
		expect(outcome.unrestorable[0]!.reason).toMatch(/sandbox workspace .* is missing/);
		expect(followUp).not.toHaveBeenCalled();
	});

	it("gives a distinct reason for a foreign owner, a live owner and a missing baseline commit", async () => {
		const { artifactsDir, sidecar, ownerFile } = await crashedIsolatedRun(true);

		await Bun.write(ownerFile, JSON.stringify({ pid: await deadPid(), id: "Other" }));
		const foreign = await adoptIsolatedSandbox(artifactsDir, id);

		await Bun.write(ownerFile, JSON.stringify({ pid: process.pid, id }));
		const live = await adoptIsolatedSandbox(artifactsDir, id);

		await Bun.write(ownerFile, JSON.stringify({ pid: await deadPid(), id }));
		const missingCommit: IsolationSidecar = {
			...sidecar,
			baseline: { ...sidecar.baseline, root: { ...sidecar.baseline.root, headCommit: "0".repeat(40) } },
		};
		await writeIsolationSidecar(artifactsDir, missingCommit);
		const gone = await adoptIsolatedSandbox(artifactsDir, id);

		const reasons = [foreign, live, gone].map(r => (r.ok ? "adopted" : r.reason));
		expect(reasons[0]).toMatch(/owned by Other/);
		expect(reasons[1]).toMatch(/live process/);
		expect(reasons[2]).toMatch(/baseline commit .* no longer exists/);
		expect(new Set(reasons).size).toBe(3);
	});

	it("re-claims the owner marker for the current process on adoption", async () => {
		const { artifactsDir, ownerFile } = await crashedIsolatedRun(true);

		const adopted = await adoptIsolatedSandbox(artifactsDir, id);

		expect(adopted.ok).toBe(true);
		expect(await Bun.file(ownerFile).json()).toMatchObject({ pid: process.pid, id });
	});

	it("adopting twice in the owning process is idempotent and lets the reviver reach the sandbox; an unadopted sandbox is refused", async () => {
		const { artifactsDir, handle, mainFile } = await crashedIsolatedRun(true);
		const subFile = path.join(mainFile.slice(0, -6), `${id}.jsonl`);
		const ref = { id, sessionFile: subFile, displayName: id, kind: "sub", parentId: MAIN_AGENT_ID };
		const factory = createPersistedSubagentReviverFactory({
			session: {} as AgentSession,
		} as unknown as Parameters<typeof createPersistedSubagentReviverFactory>[0]);

		// Cold revive outside --restore: nothing would capture the sandbox's edits.
		const warn = vi.spyOn(logger, "warn");
		expect(await factory(ref as unknown as Parameters<typeof factory>[0])).toBeUndefined();
		expect(warn).toHaveBeenCalledWith("Isolated subagent not revived", {
			id,
			reason: expect.stringMatching(/only `omp --restore` revives into its sandbox/),
		});

		const first = await adoptIsolatedSandbox(artifactsDir, id);
		// The first adoption re-claimed the owner marker for this (live) process;
		// a second adoption must not refuse it as "owned by a live process".
		const second = await adoptIsolatedSandbox(artifactsDir, id);
		expect(first.ok).toBe(true);
		expect(second).toBe(first);
		expect(await fs.stat(handle.mergedDir)).toBeDefined();

		// After restore's adoption the reviver passes its sandbox gate.
		expect(await factory(ref as unknown as Parameters<typeof factory>[0])).toBeInstanceOf(Function);
	});

	/**
	 * Mocks the restored turn so it leaves what the real one does: the agent's
	 * session live and idle in the registry, adopted with a cached reviver (the
	 * closure `ensureLive` caches survives park). Returns the session's dispose spy.
	 */
	function restoredTurnLeavesAgentLive(result: SingleResult) {
		const dispose = vi.fn(async () => {});
		const session = { dispose } as unknown as AgentSession;
		vi.spyOn(executorModule, "runSubagentFollowUpTurn").mockImplementation(async () => {
			const registry = AgentRegistry.global();
			const ref = registry.get(id)!;
			registry.attachSession(id, session, undefined, ref);
			registry.setStatus(id, "idle", ref);
			AgentLifecycleManager.global().adopt(id, { idleTtlMs: 0, revive: async () => session }, ref);
			return result;
		});
		return dispose;
	}

	// B5: without the release, a later wake (IRC, peer message) got the still-live
	// session back from ensureLive and ran a turn in the merged-and-deleted sandbox.
	it("ends a completed restore's lifecycle: ensureLive refuses instead of running in the deleted sandbox", async () => {
		const { mainFile, artifactsDir } = await crashedIsolatedRun(true);
		const dispose = restoredTurnLeavesAgentLive(turnResult);
		await restoreSession(restoringSession(mainFile));
		await manager!.getJob(id)!.promise;

		expect(getAdoptedSandbox(artifactsDir, id)).toBeUndefined();
		expect(dispose).toHaveBeenCalledTimes(1);
		expect(AgentRegistry.global().get(id)).toBeUndefined();
		await expect(AgentLifecycleManager.global().ensureLive(id)).rejects.toThrow(/Unknown agent "Iso"/);
	});

	// B5: the kept sandbox has nothing to capture later edits, so the agent must not
	// stay live (or revivable) into it either.
	it("keeps the sandbox after a failed restored turn, names it in the error and refuses later revival", async () => {
		const { mainFile, artifactsDir, handle } = await crashedIsolatedRun(true);
		const dispose = restoredTurnLeavesAgentLive({ ...turnResult, exitCode: 1, error: "boom" });
		await restoreSession(restoringSession(mainFile));
		const job = manager!.getJob(id)!;
		await job.promise;

		expect(job.status).toBe("failed");
		expect(job.errorText).toContain(handle.mergedDir);
		expect((await fs.stat(handle.mergedDir)).isDirectory()).toBe(true);
		expect(getAdoptedSandbox(artifactsDir, id)).toBeUndefined();
		expect(dispose).toHaveBeenCalledTimes(1);
		expect(AgentRegistry.global().get(id)).toBeUndefined();
		await expect(AgentLifecycleManager.global().ensureLive(id)).rejects.toThrow(/Unknown agent "Iso"/);
	});

	// C4: a revived isolated session must not expose roots outside its sandbox, as
	// the live executor's isolated run already guarantees.
	it("revives an adopted isolated agent into its sandbox without the owner's additional directories", async () => {
		const { artifactsDir, handle } = await crashedIsolatedRun(true);
		const subFile = path.join(artifactsDir, `${id}.jsonl`);
		const ref = { id, sessionFile: subFile, displayName: id, kind: "sub", parentId: MAIN_AGENT_ID };
		const extraRoot = path.join(os.tmpdir(), "omp-owner-extra-root");
		const ownerSettings = Settings.isolated({ "workspace.additionalDirectories": [extraRoot] });
		MCPManager.setInstance({ getTools: () => [], addToolsChangedListener: () => () => {} } as unknown as MCPManager);
		let captured: CreateAgentSessionOptions | undefined;
		vi.spyOn(sdkModule, "createAgentSession").mockImplementation(async options => {
			captured = options;
			throw new Error("stop after capturing the revive options");
		});
		const factory = createPersistedSubagentReviverFactory({
			session: {
				sessionManager: { getCwd: () => path.dirname(artifactsDir), getArtifactManager: () => undefined },
			} as unknown as AgentSession,
			authStorage: {} as never,
			modelRegistry: {} as never,
			settings: ownerSettings,
			enableLsp: false,
		});
		const typedRef = ref as unknown as Parameters<typeof factory>[0];

		expect((await adoptIsolatedSandbox(artifactsDir, id)).ok).toBe(true);
		const reviver = await factory(typedRef);
		if (!reviver) throw new Error("Expected a reviver for the adopted sandbox");
		await expect(reviver(typedRef)).rejects.toThrow("stop after capturing the revive options");

		expect(cfgWorkspaceAdditionalDirectories.get(ownerSettings)).toEqual([extraRoot]);
		expect(captured?.cwd).toBe(handle.mergedDir);
		expect(cfgWorkspaceAdditionalDirectories.get(captured!.settings!)).toEqual([]);
	});

	it("makes a cached reviver refuse once its restore settled, and explains a refusal outside --restore", async () => {
		const { artifactsDir, mainFile } = await crashedIsolatedRun(true);
		const subFile = path.join(mainFile.slice(0, -6), `${id}.jsonl`);
		await ensurePersistedRoster(AgentRegistry.global(), mainFile);
		const ref = AgentRegistry.global().get(id)!;
		const factory = createPersistedSubagentReviverFactory({
			session: {} as AgentSession,
		} as unknown as Parameters<typeof createPersistedSubagentReviverFactory>[0]);
		AgentLifecycleManager.global().setPersistedSubagentReviverFactory(factory, () => 0);

		// Outside --restore the caller learns why, not just "no reviver registered".
		await expect(AgentLifecycleManager.global().ensureLive(id)).rejects.toThrow(/only `omp --restore` revives/);

		await adoptIsolatedSandbox(artifactsDir, id);
		const reviver = await factory(ref);
		expect(reviver).toBeInstanceOf(Function);
		forgetAdoptedSandbox(artifactsDir, id);
		await expect(reviver!(ref)).rejects.toThrow(/restore finished/);
		expect(subFile).toContain(id);
	});

	it("hands the sandbox back as found when registering the restore job throws", async () => {
		const { mainFile, artifactsDir, ownerFile } = await crashedIsolatedRun(true);
		const before = await Bun.file(ownerFile).text();
		const session = restoringSession(mainFile);
		vi.spyOn(manager!, "register").mockImplementation(() => {
			throw new Error("register failed");
		});

		await expect(restoreSession(session)).rejects.toThrow("register failed");

		expect(getAdoptedSandbox(artifactsDir, id)).toBeUndefined();
		expect(await Bun.file(ownerFile).text()).toBe(before);
	});
});
