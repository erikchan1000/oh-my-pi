import { afterEach, describe, expect, it, vi } from "bun:test";
import type { AgentToolCall, SpeculativeOperationSink } from "@oh-my-pi/pi-agent-core";
import type { Context, Model, ProviderSessionState } from "@oh-my-pi/pi-ai";
import { createMockModel, type MockResponse } from "@oh-my-pi/pi-ai/providers/mock";
import { buildTransformedCodexRequestBody } from "@oh-my-pi/pi-ai/providers/openai-codex-responses";
import { buildParams } from "@oh-my-pi/pi-ai/providers/openai-responses";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import { parseArgs } from "@oh-my-pi/pi-coding-agent/cli/args";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { CommandController } from "@oh-my-pi/pi-coding-agent/modes/controllers/command-controller";
import { TanCommandController } from "@oh-my-pi/pi-coding-agent/modes/controllers/tan-command-controller";
import type { InteractiveModeContext } from "@oh-my-pi/pi-coding-agent/modes/types";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import * as sdk from "@oh-my-pi/pi-coding-agent/sdk";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import type { SessionHeader } from "@oh-my-pi/pi-coding-agent/session/session-entries";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import * as settingsStream from "@oh-my-pi/pi-coding-agent/session/settings-stream-fn";
import type { TerminalLaunchRequest } from "@oh-my-pi/pi-coding-agent/subprocess/terminal-launch";
import { TaskTool } from "@oh-my-pi/pi-coding-agent/task";
import { createPersistedSubagentReviverFactory } from "@oh-my-pi/pi-coding-agent/task/persisted-revive";
import { EventBus } from "@oh-my-pi/pi-coding-agent/utils/event-bus";
import * as discoveryModule from "@oh-my-pi/pi-coding-agent/task/discovery";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import { AgentOutputManager } from "@oh-my-pi/pi-coding-agent/task/output-manager";
import { TASK_FORK_ENTRY_TYPE, type TaskForkEntryData } from "@oh-my-pi/pi-coding-agent/task/fork";
import type { TaskLaunchSession } from "@oh-my-pi/pi-coding-agent/task/speculative-launch";
import type { AgentDefinition } from "@oh-my-pi/pi-coding-agent/task/types";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { TempDir } from "@oh-my-pi/pi-utils";

const muse = getBundledModel("muse-code", "muse-spark-1.3-contributor");
const luna = getBundledModel("openai-codex", "gpt-6-luna");

afterEach(() => {
	vi.restoreAllMocks();
	AgentLifecycleManager.resetGlobalForTests();
	AgentRegistry.resetGlobalForTests();
});

async function fixture() {
	const dir = TempDir.createSync("@omp-task-cache-forks-");
	const auth = await AuthStorage.create(dir.join("auth.db"));
	auth.keys.setRuntime("muse-code", "test-key");
	auth.keys.setRuntime("openai-codex", "test-key");
	const sessions: AgentSession[] = [];
	const requests: Array<{
		key: unknown;
		sessionId: string | undefined;
		state: Map<string, ProviderSessionState> | undefined;
	}> = [];
	const mock = createMockModel({ handler: { content: [{ type: "text", text: "Review complete." }] } });
	const realStreamFactory = settingsStream.createSettingsAwareStreamFn;
	vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Network is disabled in this test"));
	vi.spyOn(settingsStream, "createSettingsAwareStreamFn").mockImplementation((settings, _base, slow) =>
		realStreamFactory(
			settings,
			async (model, context, options) => {
				const cacheOptions = {
					sessionId: options?.sessionId,
					promptCacheKey: options?.promptCacheKey,
					providerSessionState: options?.providerSessionState,
				};
				const body =
					model.api === "openai-codex-responses"
						? await buildTransformedCodexRequestBody(
								model as Model<"openai-codex-responses">,
								context,
								cacheOptions,
							)
						: buildParams(model as Model<"openai-responses">, context, cacheOptions, undefined).params;
				requests.push({
					key: body.prompt_cache_key,
					sessionId: options?.sessionId,
					state: options?.providerSessionState,
				});
				return mock.stream(model, context, options);
			},
			slow,
		),
	);
	const settings = Settings.isolated({
		"async.enabled": true,
		"marketplace.autoUpdate": "off",
		"compaction.enabled": false,
		"task.enableLsp": false,
	});
	const common: sdk.CreateAgentSessionOptions = {
		cwd: dir.path(),
		agentDir: dir.path(),
		authStorage: auth,
		model: muse,
		settings,
		systemPrompt: ["Review the supplied code."],
		toolNames: [],
		disableExtensionDiscovery: true,
		skills: [],
		contextFiles: [],
		promptTemplates: [],
		rules: [],
		enableMCP: false,
		enableLsp: false,
	};
	async function create(options: sdk.CreateAgentSessionOptions = {}) {
		const { session } = await sdk.createAgentSession({ ...common, ...options });
		sessions.push(session);
		return session;
	}
	async function request(session: AgentSession, model: Model, key: string | undefined) {
		await session.setModel(model);
		await session.prompt("Continue the code review.");
		expect(requests.at(-1)?.key).toBe(key ?? session.agent.sessionId);
	}
	function context(session: AgentSession): InteractiveModeContext {
		return {
			session,
			sessionManager: session.sessionManager,
			settings,
			showError: (message: string) => {
				throw new Error(message);
			},
			showWarning: () => {},
			showStatus: () => {},
			showHookConfirm: async () => true,
			rebuildChatFromMessages: () => {},
		} as unknown as InteractiveModeContext;
	}
	return {
		dir,
		sessions,
		requests,
		create,
		request,
		context,
		async [Symbol.asyncDispose]() {
			await Promise.all(sessions.map(session => session.dispose()));
			auth.close();
			dir[Symbol.dispose]();
		},
	};
}

async function persistedHeader(manager: SessionManager): Promise<SessionHeader> {
	await manager.ensureOnDisk();
	await manager.flush();
	const file = manager.getSessionFile();
	if (!file) throw new Error("Expected a persisted session");
	const records = Bun.JSONL.parse(await Bun.file(file).text()) as Array<{ type: string }>;
	const header = records.find(record => record.type === "session");
	if (!header) throw new Error("Missing session header");
	return header as SessionHeader;
}

it.each([false, true])(
	"preserves automatic versus explicit routing through live and terminal forks (explicit=%s)",
	async explicit => {
		await using h = await fixture();
		const manager = SessionManager.create(h.dir.path(), h.dir.join("parent"));
		const parent = await h.create({
			sessionManager: manager,
			taskPromptCacheKey: "parent:task",
			providerPromptCacheKey: explicit ? "explicit-cache" : undefined,
		});
		await h.request(parent, muse, explicit ? "explicit-cache" : "parent:task");
		const before = h.requests.at(-1)!;
		const launches: TerminalLaunchRequest[] = [];
		const controller = new CommandController(h.context(parent), {
			environment: () => ({ TMUX: "server,1,0", TMUX_PANE: "%1" }),
			resolveCliEntryCmd: () => ["omp-entry"],
			getActiveProfile: () => undefined,
			argv: () => [],
			launchTerminal: async request => {
				launches.push(request);
				return { multiplexer: request.multiplexer, placement: request.placement };
			},
		});
		await controller.handleForkCommand("pane");
		expect(launches).toHaveLength(1);
		const command = launches[0]!.command;
		const parsed = parseArgs(command.slice(command.indexOf("omp-entry") + 1));
		expect(parsed.providerPromptCacheKey).toBe(explicit ? "explicit-cache" : undefined);
		const childManager = await SessionManager.forkFrom(
			manager.getSessionFile()!,
			h.dir.path(),
			h.dir.join("terminal"),
		);
		const child = await h.create({
			sessionManager: childManager,
			providerPromptCacheKey: parsed.providerPromptCacheKey,
		});
		await h.request(child, muse, explicit ? "explicit-cache" : "parent:task");
		expect(h.requests.at(-1)?.sessionId).not.toBe(before.sessionId);
		expect(h.requests.at(-1)?.state).not.toBe(before.state);
		await h.request(child, luna, explicit ? "explicit-cache" : undefined);
		await h.request(child, muse, explicit ? "explicit-cache" : "parent:task");
		expect(await parent.fork(undefined, { requireIdle: true })).toBe(true);
		await h.request(parent, muse, explicit ? "explicit-cache" : "parent:task");
		expect(h.requests.at(-1)?.sessionId).not.toBe(before.sessionId);
		await h.request(parent, luna, explicit ? "explicit-cache" : undefined);
		await h.request(parent, muse, explicit ? "explicit-cache" : "parent:task");
		for (const current of [manager, childManager]) {
			const header = await persistedHeader(current);
			expect(header.taskPromptCacheKey).toBe("parent:task");
			expect(header.providerPromptCacheKey).toBeUndefined();
		}
	},
);

it.each([false, true])("keeps tan's copied candidate model-gated (explicit=%s)", async explicit => {
	await using h = await fixture();
	const parent = await h.create({
		sessionManager: SessionManager.create(h.dir.path(), h.dir.join("parent")),
		taskPromptCacheKey: "parent:task",
		providerPromptCacheKey: explicit ? "explicit-cache" : undefined,
	});
	await h.request(parent, muse, explicit ? "explicit-cache" : "parent:task");
	const parentRequest = h.requests.at(-1)!;
	const realCreate = sdk.createAgentSession;
	vi.spyOn(sdk, "createAgentSession").mockImplementation(async options => {
		const created = await realCreate(options);
		h.sessions.push(created.session);
		await h.request(created.session, muse, explicit ? "explicit-cache" : "parent:task");
		expect(h.requests.at(-1)?.sessionId).not.toBe(parentRequest.sessionId);
		expect(h.requests.at(-1)?.state).not.toBe(parentRequest.state);
		await h.request(created.session, luna, explicit ? "explicit-cache" : undefined);
		await h.request(created.session, muse, explicit ? "explicit-cache" : "parent:task");
		return created;
	});
	const manager = parent.asyncJobManager;
	if (!manager) throw new Error("Missing async job manager");
	let runJob: (() => Promise<unknown>) | undefined;
	vi.spyOn(manager, "register").mockImplementation((_type, _label, run) => {
		runJob = () =>
			run({
				jobId: "tan-test",
				signal: new AbortController().signal,
				reportProgress: async () => {},
				markRunning: () => {},
			});
		return "tan-test";
	});
	await new TanCommandController(h.context(parent)).start("Review another file");
	if (!runJob) throw new Error("Tan did not register its job");
	await runJob();
	expect(h.requests).toHaveLength(5);
});

it("restores automatic affinity from disk without promoting a legacy fork fallback", async () => {
	await using h = await fixture();
	const manager = SessionManager.create(h.dir.path(), h.dir.join("parent"));
	const seed = await h.create({ sessionManager: manager });
	await h.request(seed, muse, undefined);
	await seed.fork();
	const inherited = manager.getHeader()?.providerPromptCacheKey;
	expect(inherited).toBeDefined();
	await seed.dispose();
	const reopened = await SessionManager.open(manager.getSessionFile()!, h.dir.path());
	const seeded = await h.create({
		sessionManager: reopened,
		taskPromptCacheKey: "restored:task",
		providerPromptCacheKey: inherited,
		providerPromptCacheKeySource: "fork",
	});
	await h.request(seeded, muse, "restored:task");
	expect((await persistedHeader(reopened)).providerPromptCacheKey).toBeUndefined();
	await seeded.dispose();
	const restored = await h.create({
		sessionManager: await SessionManager.open(reopened.getSessionFile()!, h.dir.path()),
	});
	await h.request(restored, muse, "restored:task");
	await h.request(restored, luna, undefined);
	await h.request(restored, muse, "restored:task");
	await restored.dispose();
	const overridden = await h.create({
		sessionManager: await SessionManager.open(reopened.getSessionFile()!, h.dir.path()),
		providerPromptCacheKey: "caller-override",
	});
	await h.request(overridden, muse, "caller-override");
	await h.request(overridden, luna, "caller-override");
});

it("clears automatic affinity on new and partial sessions and restores the destination on switch", async () => {
	await using h = await fixture();
	const session = await h.create({
		sessionManager: SessionManager.create(h.dir.path(), h.dir.join("parent")),
		taskPromptCacheKey: "parent:task",
	});
	await h.request(session, muse, "parent:task");
	const originalFile = session.sessionManager.getSessionFile()!;
	expect(await session.newSession()).toBe(true);
	await h.request(session, muse, undefined);
	expect(await session.switchSession(originalFile)).toBe(true);
	await h.request(session, muse, "parent:task");
	const leaf = session.sessionManager.getBranch().find(entry => entry.type === "message");
	if (!leaf) throw new Error("Missing message to fork");
	expect(await session.fork(leaf.id)).toBe(true);
	await h.request(session, muse, undefined);
	expect((await persistedHeader(session.sessionManager)).taskPromptCacheKey).toBeUndefined();
});

// ---------------------------------------------------------------------------
// Forked-context task spawns (`fork: true`)
// ---------------------------------------------------------------------------

const forkTaskAgent: AgentDefinition = {
	name: "task",
	description: "General-purpose task agent",
	systemPrompt: "Do the assigned work.",
	source: "bundled",
};
const FORK_CALL_ID = "call-fork-1";
const forkItem = { name: "ForkChild", task: "Review module A.", solutionSpace: "one fix: rename", fork: true };

function toolResultText(message: unknown): string {
	return JSON.stringify((message as { content?: unknown }).content);
}

/** Parent session whose first turn calls `task`; `onChild` scripts the forked child's turns. */
async function forkFixture(options: {
	cwd?: string;
	settings?: Record<string, unknown>;
	taskArgs?: Record<string, unknown>;
	onChild?: (context: Context) => MockResponse;
	sessionOptions?: Partial<sdk.CreateAgentSessionOptions>;
}) {
	const dir = TempDir.createSync("@omp-task-fork-");
	const auth = await AuthStorage.create(dir.join("auth.db"));
	auth.keys.setRuntime("muse-code", "test-key");
	const bodies: Array<{ key: unknown; model: string }> = [];
	const taskArgs = options.taskArgs ?? { context: "Shared context.", tasks: [forkItem] };
	const mock = createMockModel({
		handler: context => {
			// Only a cold (non-fork) child runs under the task agent's own role prompt.
			if (JSON.stringify(context.systemPrompt).includes(forkTaskAgent.systemPrompt)) {
				return { content: ["Cold child done."] };
			}
			const last = context.messages.at(-1);
			if (last?.role === "toolResult") {
				const text = toolResultText(last);
				if (text.includes("forked task agent")) {
					return options.onChild ? options.onChild(context) : { content: ["Fork review done."] };
				}
				if (last.toolCallId === FORK_CALL_ID) return { content: ["Parent done."] };
				return { content: ["Fork child follow-up done."] };
			}
			return { content: [{ type: "toolCall", id: FORK_CALL_ID, name: "task", arguments: taskArgs }] };
		},
	});
	const realStreamFactory = settingsStream.createSettingsAwareStreamFn;
	vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Network is disabled in this test"));
	vi.spyOn(settingsStream, "createSettingsAwareStreamFn").mockImplementation((settings, _base, slow) =>
		realStreamFactory(
			settings,
			async (model, context, streamOptions) => {
				const body = buildParams(
					model as Model<"openai-responses">,
					context,
					{
						sessionId: streamOptions?.sessionId,
						promptCacheKey: streamOptions?.promptCacheKey,
						providerSessionState: streamOptions?.providerSessionState,
					},
					undefined,
				).params;
				bodies.push({ key: body.prompt_cache_key, model: model.id });
				return mock.stream(model, context, streamOptions);
			},
			slow,
		),
	);
	vi.spyOn(discoveryModule, "discoverAgents").mockResolvedValue({ agents: [forkTaskAgent], projectAgentsDir: null });
	const settings = Settings.isolated({
		"async.enabled": false,
		"marketplace.autoUpdate": "off",
		"compaction.enabled": false,
		"task.enableLsp": false,
		"task.batch": true,
		"task.isolation.enabled": false,
		"todo.enabled": false,
		"task.forkEnabled": true,
		...options.settings,
	});
	const { session } = await sdk.createAgentSession({
		cwd: options.cwd ?? dir.path(),
		agentDir: dir.path(),
		authStorage: auth,
		model: muse,
		settings,
		sessionManager: SessionManager.create(options.cwd ?? dir.path(), dir.join("parent")),
		systemPrompt: ["Review the supplied code."],
		toolNames: ["task"],
		disableExtensionDiscovery: true,
		skills: [],
		contextFiles: [],
		promptTemplates: [],
		rules: [],
		enableMCP: false,
		enableLsp: false,
		...options.sessionOptions,
	});
	return {
		dir,
		session,
		mock,
		bodies,
		async [Symbol.asyncDispose]() {
			await session.dispose();
			auth.close();
			dir[Symbol.dispose]();
		},
	};
}

function taskResultText(session: AgentSession): string {
	const result = session.sessionManager
		.getBranch()
		.flatMap(entry => (entry.type === "message" && entry.message.role === "toolResult" ? [entry.message] : []))
		.find(message => message.toolCallId === FORK_CALL_ID);
	if (!result) throw new Error("Missing task tool result");
	return toolResultText(result);
}

it("branches the parent into a fork child that shares its prompt cache shape", async () => {
	await using h = await forkFixture({});
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	expect(h.mock.calls).toHaveLength(3);
	const [parentCall, childCall] = h.mock.calls;
	const parentContext = parentCall!.context;
	const childContext = childCall!.context;
	// (a) exact parent prefix up to the task call.
	expect(childContext.messages.slice(0, parentContext.messages.length)).toEqual(parentContext.messages);
	const assistant = childContext.messages[parentContext.messages.length]!;
	expect(assistant.role).toBe("assistant");
	expect(JSON.stringify(assistant)).toContain(FORK_CALL_ID);
	// (b) parent's exact system prompt, tools, model and prompt cache key.
	expect(childContext.systemPrompt).toEqual(parentContext.systemPrompt);
	// Full definitions (name, description, parameter schema, order): prompt-cache reuse needs byte-identical tools.
	expect(toolShape(childContext)).toEqual(toolShape(parentContext));
	expect(h.bodies[1]!.key).toBe(h.bodies[0]!.key);
	expect(h.bodies[1]!.model).toBe(h.bodies[0]!.model);
	// (c) ends with the seed tool result carrying the child id and task text.
	const seed = childContext.messages.at(-1)!;
	expect(seed.role).toBe("toolResult");
	expect((seed as { toolCallId: string }).toolCallId).toBe(FORK_CALL_ID);
	expect(toolResultText(seed)).toContain("ForkChild");
	expect(toolResultText(seed)).toContain("Review module A.");
	// Identity lives only in the tool result, never the system prompt.
	expect(JSON.stringify(childContext.systemPrompt)).not.toContain("ForkChild");
	expect(taskResultText(h.session)).toContain("Fork review done.");
});

it("tells a fork child that replays the parent's pending task call to focus on its own task", async () => {
	await using h = await forkFixture({
		onChild: () => ({
			content: [
				{ type: "toolCall", id: "child-call-1", name: "task", arguments: { context: "c", tasks: [forkItem] } },
			],
		}),
	});
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	const guarded = h.mock.calls.find(
		call =>
			call.context.messages.at(-1)?.role === "toolResult" &&
			toolResultText(call.context.messages.at(-1)).includes("focus on your own"),
	);
	if (!guarded) throw new Error("Expected the replay guard result to reach the child");
	expect(toolResultText(guarded.context.messages.at(-1))).toContain("confused");
});

it("refuses fork spawns above task.forkMaxParentTokens with an explicit error", async () => {
	await using h = await forkFixture({ settings: { "task.forkMaxParentTokens": 1 } });
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	const text = taskResultText(h.session);
	expect(text).toContain("task.forkMaxParentTokens");
	expect(text).toContain("(1)");
	// The refused call never spawned a cold child.
	expect(h.mock.calls).toHaveLength(2);
});

it("refuses fork spawns when task.forkEnabled is off", async () => {
	await using h = await forkFixture({ settings: { "task.forkEnabled": false } });
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	expect(taskResultText(h.session)).toContain("task.forkEnabled");
	expect(h.mock.calls).toHaveLength(2);
});

/** Provider-visible tool definitions: name, description and serialized parameter schema, in order. */
function toolShape(context: Context) {
	return (context.tools ?? []).map(tool => ({
		name: tool.name,
		description: tool.description,
		parameters: JSON.stringify(tool.parameters),
	}));
}

function taskToolDescription(context: Context): string {
	const tool = context.tools?.find(candidate => candidate.name === "task");
	if (!tool) throw new Error("Parent request is missing the task tool");
	return tool.description;
}

it("documents `fork` in the task tool description when task.forkEnabled is on", async () => {
	await using on = await forkFixture({});
	await on.session.prompt("Investigate the repo.");
	await on.session.waitForIdle();
	expect(taskToolDescription(on.mock.calls[0]!.context)).toContain("shares its prompt cache");
});

it("omits `fork` from the task tool description when task.forkEnabled is off", async () => {
	await using off = await forkFixture({
		settings: { "task.forkEnabled": false },
		taskArgs: { context: "Shared context.", tasks: [{ name: "Cold", task: "Review module B." }] },
	});
	await off.session.prompt("Investigate the repo.");
	await off.session.waitForIdle();
	const description = taskToolDescription(off.mock.calls[0]!.context);
	expect(description).not.toContain("`fork`");
	expect(description).not.toContain("shares its prompt cache");
});

it("keeps non-fork spawns cold: fresh transcript, agent role prompt, no seed", async () => {
	await using h = await forkFixture({
		taskArgs: { context: "Shared context.", tasks: [{ name: "Cold", task: "Review module B." }] },
	});
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	expect(h.mock.calls).toHaveLength(3);
	const [parentCall, childCall] = h.mock.calls;
	const childContext = childCall!.context;
	expect(childContext.systemPrompt).not.toEqual(parentCall!.context.systemPrompt);
	expect(JSON.stringify(childContext.systemPrompt)).toContain(forkTaskAgent.systemPrompt);
	expect(childContext.messages.some(message => message.role === "toolResult")).toBe(false);
	expect(JSON.stringify(childContext.messages)).not.toContain("Investigate the repo.");
	expect(JSON.stringify(childContext.messages)).not.toContain("forked task agent");
	expect(JSON.stringify(childContext.messages)).toContain("Review module B.");
	expect(taskResultText(h.session)).toContain("Cold child done.");
});

it("runs a fork child in an isolated worktree", async () => {
	const repo = TempDir.createSync("@omp-task-fork-repo-");
	try {
		for (const args of [
			["init", "-q"],
			["config", "user.email", "t@example.com"],
			["config", "user.name", "t"],
		]) {
			await Bun.spawn(["git", ...args], { cwd: repo.path() }).exited;
		}
		await Bun.write(repo.join("a.txt"), "a\n");
		await Bun.spawn(["git", "add", "."], { cwd: repo.path() }).exited;
		await Bun.spawn(["git", "commit", "-q", "-m", "init"], { cwd: repo.path() }).exited;
		await using h = await forkFixture({
			cwd: repo.path(),
			settings: { "task.isolation.enabled": true },
			taskArgs: { context: "Shared context.", tasks: [{ ...forkItem, isolated: true }] },
		});
		await h.session.prompt("Investigate the repo.");
		await h.session.waitForIdle();
		expect(h.mock.calls).toHaveLength(3);
		expect(taskResultText(h.session)).toContain("Fork review done.");
	} finally {
		repo.removeSync();
	}
});

it("forks from a depth-1 parent: the child keeps `task` and the recursion cap is enforced at execution", async () => {
	await using h = await forkFixture({
		sessionOptions: { taskDepth: 1 },
		onChild: () => ({
			content: [
				{
					type: "toolCall",
					id: "child-call-1",
					name: "task",
					arguments: {
						context: "c",
						tasks: [{ name: "Other", task: "Something unrelated.", solutionSpace: "one fix" }],
					},
				},
			],
		}),
	});
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	const [parentCall, childCall] = h.mock.calls;
	expect(toolShape(childCall!.context)).toEqual(toolShape(parentCall!.context));
	expect(childCall!.context.tools?.some(tool => tool.name === "task")).toBe(true);
	// Depth 2 with task.maxRecursionDepth=2: the depth verdict is reachable and returned as a tool result.
	const guarded = h.mock.calls.find(
		call =>
			call.context.messages.at(-1)?.role === "toolResult" &&
			toolResultText(call.context.messages.at(-1)).includes("task.maxRecursionDepth"),
	);
	if (!guarded) throw new Error("Expected the depth refusal to reach the fork child");
});

it("refuses a fork when the parent holds tools a child cannot share byte for byte", async () => {
	await using h = await forkFixture({
		sessionOptions: {
			customTools: [
				{
					name: "parity_probe",
					label: "Parity probe",
					description: "A custom tool outside the builtin set.",
					parameters: { type: "object", properties: {}, additionalProperties: false } as never,
					execute: async () => ({ content: [{ type: "text" as const, text: "ok" }] }),
				},
			],
		},
	});
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	const text = taskResultText(h.session);
	expect(text).toContain("parity_probe");
	expect(text).toContain("`fork: true` refused");
	// No child session was built.
	expect(h.mock.calls).toHaveLength(2);
});

it.each([
	["effort", { effort: "lo" }],
	["outputSchema", { outputSchema: { type: "object" } }],
])("refuses fork combined with %s instead of silently overriding it", async (field, extra) => {
	await using h = await forkFixture({
		taskArgs: { context: "Shared context.", tasks: [{ ...forkItem, ...extra }] },
		settings: { "task.enableEffort": true, "task.structured": true },
	});
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	const text = taskResultText(h.session);
	expect(text).toContain(`\`${field}\``);
	expect(h.mock.calls).toHaveLength(2);
});

it("refuses a fork when the parent session has no agent id instead of spawning a cold child", async () => {
	await using h = await forkFixture({ sessionOptions: { agentId: "" } });
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	const text = taskResultText(h.session);
	expect(text).toContain("`fork: true` refused");
	expect(text).toContain("no agent id");
	expect(h.mock.calls).toHaveLength(2);
});

it("does not leak a fork spec into a later cold spawn from the same parent", async () => {
	const taskArgs: { context: string; tasks: object[] } = { context: "Shared context.", tasks: [forkItem] };
	await using h = await forkFixture({ taskArgs });
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	expect(h.mock.calls).toHaveLength(3);
	taskArgs.tasks = [{ name: "Cold", task: "Review module B." }];
	await h.session.prompt("Now review module B.");
	await h.session.waitForIdle();
	const coldCall = h.mock.calls.find(call =>
		JSON.stringify(call.context.systemPrompt).includes(forkTaskAgent.systemPrompt),
	);
	if (!coldCall) throw new Error("Expected the second spawn to run as a cold child");
	expect(coldCall.context.messages.some(message => message.role === "toolResult")).toBe(false);
	expect(JSON.stringify(coldCall.context.messages)).not.toContain("forked task agent");
	expect(JSON.stringify(coldCall.context.messages)).not.toContain("Investigate the repo.");
});

it.each([
	["an explicit ask/read tool list", { toolNames: ["task", "ask", "read"] }],
	["the real default tool set", { toolNames: undefined }],
])("forks from an interactive parent (hasUI) with %s and keeps the tool list identical", async (_label, extra) => {
	await using h = await forkFixture({ sessionOptions: { hasUI: true, ...extra } });
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	expect(h.mock.calls).toHaveLength(3);
	const [parentCall, childCall] = h.mock.calls;
	expect(parentCall!.context.tools?.some(tool => tool.name === "ask")).toBe(true);
	expect(toolShape(childCall!.context)).toEqual(toolShape(parentCall!.context));
	expect(taskResultText(h.session)).toContain("Fork review done.");
});

it("refuses a fork before spawning when the child-shaped tool plan would drop a parent tool", async () => {
	await using h = await forkFixture({ settings: { "bash.enabled": false } });
	// The parent reports `bash` as enabled while the setting gates it off, so a child built from the
	// same settings would not receive it. The pre-check must refuse instead of building the child.
	const enabled = h.session.getEnabledToolNames.bind(h.session);
	vi.spyOn(h.session, "getEnabledToolNames").mockImplementation(() => [...enabled(), "bash"]);
	await h.session.prompt("Investigate the repo.");
	await h.session.waitForIdle();
	const text = taskResultText(h.session);
	expect(text).toContain("`fork: true` refused");
	expect(text).toContain("would not receive these parent tools: bash");
	// Refused before spawn: no child request and no post-spawn drift failure.
	expect(text).not.toContain("differ from the parent");
	expect(h.mock.calls).toHaveLength(2);
});

// Speculative launch of a fork child's own streamed `task` call (round-2 C1): a call the replay or
// depth guard refuses at dispatch must never start running ahead of dispatch.

const replayedTask = forkItem.task;
const speculativeSink: SpeculativeOperationSink = {
	maxInFlight: 2,
	admit: async () => undefined,
	authorizeLaunch: async () => ({ allowed: true }),
	close: () => {},
};

/** A fork child's tool session: its branch carries the inherited `task_fork` entry. */
function forkChildToolSession(manager: AsyncJobManager, taskDepth: number): ToolSession {
	const data: TaskForkEntryData = {
		forkedFromToolCallId: FORK_CALL_ID,
		ancestorPendingToolCallIds: [FORK_CALL_ID],
		ancestorPendingTasks: [replayedTask],
	};
	const entry = {
		type: "custom",
		customType: TASK_FORK_ENTRY_TYPE,
		data,
		id: "fork-entry",
		parentId: null,
		timestamp: "2026-01-01T00:00:00.000Z",
	};
	return {
		cwd: "/tmp",
		hasUI: false,
		settings: Settings.isolated({ "async.enabled": true, "task.batch": true }),
		getSessionFile: () => null,
		getSessionSpawns: () => "*",
		getAgentId: () => null,
		asyncJobManager: manager,
		forkOfParent: true,
		taskDepth,
		sessionManager: { getBranch: () => [entry] },
	} as unknown as ToolSession;
}

it.each([
	["replays a task pending in the parent prefix", 0, replayedTask, false],
	["exceeds task.maxRecursionDepth", 2, "Unrelated follow-up work.", false],
	["is a fresh task with recursion headroom (control)", 0, "Unrelated follow-up work.", true],
])("speculative launch from a fork child whose streamed call %s", async (_label, taskDepth, text, launches) => {
	const manager = new AsyncJobManager({ onJobComplete: () => {} });
	vi.spyOn(discoveryModule, "discoverAgents").mockResolvedValue({ agents: [forkTaskAgent], projectAgentsDir: null });
	// Claiming an agent id is the last step before a speculative run starts.
	const allocate = vi.spyOn(AgentOutputManager.prototype, "allocate");
	vi.spyOn(executorModule, "runSubprocess").mockImplementation(async options => {
		const aborted = Promise.withResolvers<void>();
		if (options.signal?.aborted) aborted.resolve();
		options.signal?.addEventListener("abort", () => aborted.resolve(), { once: true });
		await aborted.promise;
		throw new Error("speculative run aborted");
	});
	try {
		const tool = await TaskTool.create(forkChildToolSession(manager, taskDepth));
		const toolCall: AgentToolCall = { type: "toolCall", id: "child-call-1", name: "task", arguments: {} };
		const session = (await tool.speculation.stream?.open({
			coordinator: speculativeSink,
			parentToolCallId: toolCall.id,
		})) as TaskLaunchSession | undefined;
		if (!session) throw new Error("expected a task launch session");
		session.update(toolCall, JSON.stringify({ context: "c", tasks: [{ name: "Again", task: text }] }));
		// `adopt` waits for every queued launch to settle; adopting nothing then discards whatever started.
		expect((await session.adopt([])).size).toBe(0);
		expect(allocate).toHaveBeenCalledTimes(launches ? 1 : 0);
		if (!launches) expect(executorModule.runSubprocess).not.toHaveBeenCalled();
	} finally {
		await manager.dispose({ timeoutMs: 1000 });
	}
});

describe("persisted fork child revival", () => {
	async function revivableFork(h: Awaited<ReturnType<typeof forkFixture>>) {
		await h.session.prompt("Investigate the repo.");
		await h.session.waitForIdle();
		const childRef = AgentRegistry.global()
			.list()
			.find(ref => ref.id === "ForkChild");
		if (!childRef?.sessionFile) throw new Error("Expected the fork child to be registered with a session file");
		const factory = createPersistedSubagentReviverFactory({
			session: h.session,
			authStorage: h.session.modelRegistry.authStorage,
			modelRegistry: h.session.modelRegistry,
			settings: h.session.settings,
			enableLsp: false,
			eventBus: new EventBus(),
		});
		// Park the live child first so the revive re-creates it from its persisted contract.
		await childRef.session?.dispose();
		AgentRegistry.global().unregister(childRef.id);
		const ref = AgentRegistry.global().register({
			id: childRef.id,
			displayName: childRef.displayName,
			kind: childRef.kind,
			parentId: childRef.parentId,
			session: null,
			sessionFile: childRef.sessionFile,
			status: "parked",
		});
		return { factory, ref, sessionFile: childRef.sessionFile };
	}

	it("keeps `task` and the parent's tool shape when a fork child at the depth limit is cold revived", async () => {
		// maxRecursionDepth 1: the fork child sits at the limit, where a cold child would lose `task`.
		await using h = await forkFixture({ settings: { "task.maxRecursionDepth": 1 } });
		const { factory, ref } = await revivableFork(h);
		const parentTools = h.mock.calls[0]!.context.tools?.map(tool => tool.name);
		const reviver = await factory(ref);
		if (!reviver) throw new Error("Expected a persisted reviver for the fork child");
		const revived = await reviver(ref);
		try {
			expect(revived.getEnabledToolNames().sort()).toEqual([...(parentTools ?? [])].sort());
			expect(revived.getEnabledToolNames()).toContain("task");
		} finally {
			await revived.dispose();
		}
	});

	it("refuses to revive a persisted fork child that predates the fork contract", async () => {
		await using h = await forkFixture({ settings: { "task.maxRecursionDepth": 1 } });
		const { factory, ref, sessionFile } = await revivableFork(h);
		const text = await Bun.file(sessionFile).text();
		expect(text).toContain('"forkOfParent":true');
		await Bun.write(sessionFile, text.replaceAll('"forkOfParent":true,', "").replaceAll(',"forkOfParent":true', ""));
		const reviver = await factory(ref);
		if (!reviver) throw new Error("Expected a persisted reviver for the fork child");
		await expect(reviver(ref)).rejects.toThrow("predates the recorded fork contract");
	});
});
