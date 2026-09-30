import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  AdapterEnvironmentCheck,
  AdapterEnvironmentTestContext,
  AdapterEnvironmentTestResult,
  AdapterExecutionContext,
  AdapterExecutionResult,
} from "@paperclipai/adapter-utils";
import {
  ensureAdapterExecutionTargetCommandResolvable,
  readAdapterExecutionTarget,
  resolveAdapterExecutionTargetCwd,
} from "@paperclipai/adapter-utils/execution-target";
import {
  DEFAULT_ACP_ENGINE_MODE,
  DEFAULT_ACP_ENGINE_NON_INTERACTIVE_PERMISSIONS,
  DEFAULT_ACP_ENGINE_PERMISSION_MODE,
  DEFAULT_ACP_ENGINE_WARM_HANDLE_IDLE_MS,
} from "@paperclipai/adapter-utils/acpx-engine/constants";
import type { AcpxEngineExecutorOptions } from "@paperclipai/adapter-utils/acpx-engine/execute";
import {
  asNumber,
  asString,
  parseObject,
} from "@paperclipai/adapter-utils/server-utils";
import { DEFAULT_KIMCHI_LOCAL_MODEL } from "../index.js";

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const packageRootDir = path.resolve(moduleDir, "../..");
const MIN_ACP_NODE_VERSION = "20.0.0";

export type KimchiExecutionEngine = "acp";

export interface KimchiEngineSelection {
  engine: KimchiExecutionEngine;
  explicit: boolean;
  unavailableReason?: string;
}

type KimchiEngineResolutionInput =
  Pick<AdapterExecutionContext, "config"> &
  Partial<Pick<AdapterExecutionContext, "executionTarget" | "executionTransport">>;

type KimchiAcpExecutorOptions = Omit<
  AcpxEngineExecutorOptions,
  "adapterType" | "moduleDir" | "packageRootDir"
>;

type KimchiAcpExecutor = (ctx: AdapterExecutionContext) => Promise<AdapterExecutionResult>;

/**
 * Kimchi is ACP-only in v1. The engine key is still normalized so a stored
 * `engine` config value (including a leftover `cli` from a copied kimi
 * config) cannot route the run onto an unverified CLI-lane path.
 */
function normalizeEngine(value: unknown): KimchiEngineSelection {
  const raw = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (raw === "acp") return { engine: "acp", explicit: true };
  return { engine: "acp", explicit: false };
}

export function resolveKimchiExecutionEngine(config: Record<string, unknown>): KimchiEngineSelection {
  return normalizeEngine(config.engine);
}

export async function resolveKimchiExecutionEngineForRun(
  input: KimchiEngineResolutionInput,
): Promise<KimchiEngineSelection> {
  const selection = normalizeEngine(input.config.engine);
  const reason = await kimchiAcpUnavailableReason(input);
  return reason ? { ...selection, unavailableReason: reason } : selection;
}

function firstNonEmptyString(...values: unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const trimmed = value.trim();
    if (trimmed.length > 0) return trimmed;
  }
  return undefined;
}

export function buildKimchiAcpConfig(config: Record<string, unknown>): Record<string, unknown> {
  const configuredAgentCommand = firstNonEmptyString(config.agentCommand, config.acpAgentCommand);
  const configuredKimchiCommand = firstNonEmptyString(config.command);
  // Kimchi exposes its ACP server through the `--mode acp` flag, not a
  // subcommand (contrast kimi's `kimi acp`).
  const agentCommand = configuredAgentCommand ?? (configuredKimchiCommand ? `${configuredKimchiCommand} --mode acp` : undefined);
  const stateDir = firstNonEmptyString(config.stateDir, config.acpStateDir);
  const mode = firstNonEmptyString(config.mode, config.acpMode) ?? DEFAULT_ACP_ENGINE_MODE;
  const permissionMode =
    firstNonEmptyString(config.permissionMode, config.acpPermissionMode) ??
    DEFAULT_ACP_ENGINE_PERMISSION_MODE;
  const nonInteractivePermissions =
    firstNonEmptyString(config.nonInteractivePermissions, config.acpNonInteractivePermissions) ??
    DEFAULT_ACP_ENGINE_NON_INTERACTIVE_PERMISSIONS;
  const warmHandleIdleMs =
    config.warmHandleIdleMs ??
    config.acpWarmHandleIdleMs ??
    DEFAULT_ACP_ENGINE_WARM_HANDLE_IDLE_MS;

  const next: Record<string, unknown> = {
    ...config,
    agent: "kimchi",
    mode,
    permissionMode,
    nonInteractivePermissions,
    warmHandleIdleMs,
    ...(agentCommand ? { agentCommand } : {}),
    ...(stateDir ? { stateDir } : {}),
  };
  const model = asString(next.model, "").trim();
  if (!model || model === DEFAULT_KIMCHI_LOCAL_MODEL) delete next.model;
  return next;
}

function withKimchiAcpDefaults(options: KimchiAcpExecutorOptions): AcpxEngineExecutorOptions {
  return {
    ...options,
    adapterType: "kimchi_local",
    moduleDir,
    packageRootDir,
  };
}

export function createKimchiAcpExecutor(options: KimchiAcpExecutorOptions = {}): KimchiAcpExecutor {
  let executor: KimchiAcpExecutor | null = null;
  return async (ctx) => {
    let currentExecutor = executor;
    if (!currentExecutor) {
      const { createAcpxEngineExecutor } = await import("@paperclipai/adapter-utils/acpx-engine/execute");
      currentExecutor = createAcpxEngineExecutor(withKimchiAcpDefaults(options));
      executor = currentExecutor;
    }
    return currentExecutor({
      ...ctx,
      config: buildKimchiAcpConfig(ctx.config),
    });
  };
}

function parseVersion(version: string): [number, number, number] {
  const match = version.match(/^v?(\d+)\.(\d+)\.(\d+)/);
  if (!match) return [0, 0, 0];
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

export function nodeVersionMeetsKimchiAcpMinimum(version = process.version): boolean {
  const [major, minor, patch] = parseVersion(version);
  const [minMajor, minMinor, minPatch] = parseVersion(MIN_ACP_NODE_VERSION);
  if (major !== minMajor) return major > minMajor;
  if (minor !== minMinor) return minor > minMinor;
  return patch >= minPatch;
}

async function pathExists(candidate: string): Promise<boolean> {
  return fs.access(candidate).then(() => true).catch(() => false);
}

function hasPathSeparator(command: string): boolean {
  return command.includes("/") || command.includes("\\");
}

function firstShellToken(command: string): string | null {
  const trimmed = command.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith("'") || trimmed.startsWith("\"")) return null;
  return trimmed.split(/\s+/, 1)[0] ?? null;
}

async function findCommandOnPath(binName: string, pathValue = process.env.PATH ?? ""): Promise<string | null> {
  for (const segment of pathValue.split(path.delimiter)) {
    if (!segment) continue;
    const candidate = path.join(segment, binName);
    if (await pathExists(candidate)) return candidate;
  }
  return null;
}

function resolveConfigPath(config: Record<string, unknown>): string {
  const envConfig = parseObject(config.env);
  return typeof envConfig.PATH === "string" && envConfig.PATH.trim().length > 0
    ? envConfig.PATH
    : process.env.PATH ?? "";
}

async function commandIsResolvable(
  command: string,
  pathValue = process.env.PATH ?? "",
  input?: KimchiEngineResolutionInput,
): Promise<boolean> {
  const token = firstShellToken(command);
  if (!token) return true;
  const target = readAdapterExecutionTarget({
    executionTarget: input?.executionTarget,
    legacyRemoteExecution: input?.executionTransport?.remoteExecution,
  });
  if (target?.kind === "remote") {
    try {
      await ensureAdapterExecutionTargetCommandResolvable(
        token,
        target,
        resolveAdapterExecutionTargetCwd(target, asString(input?.config.cwd, ""), process.cwd()),
        process.env,
      );
      return true;
    } catch {
      return false;
    }
  }
  if (path.isAbsolute(token) || hasPathSeparator(token)) return pathExists(token);
  return (await findCommandOnPath(token, pathValue)) !== null;
}

function resolveKimchiAcpCommand(config: Record<string, unknown>): string {
  const configured = firstNonEmptyString(config.agentCommand, config.acpAgentCommand);
  if (configured) return configured;
  const kimchiCommand = firstNonEmptyString(config.command) ?? "kimchi";
  return `${kimchiCommand} --mode acp`;
}

function sandboxTargetHasProcessSessionBridge(
  target: ReturnType<typeof readAdapterExecutionTarget>,
): boolean {
  return target?.kind === "remote" && target.transport === "sandbox" && Boolean(target.runner);
}

async function kimchiAcpUnavailableReason(
  input: KimchiEngineResolutionInput,
): Promise<string | null> {
  const target = readAdapterExecutionTarget({
    executionTarget: input.executionTarget,
    legacyRemoteExecution: input.executionTransport?.remoteExecution,
  });
  if (target?.kind === "remote" && !sandboxTargetHasProcessSessionBridge(target)) {
    if (target.transport === "sandbox") {
      return "Kimchi ACP requires a bidirectional remote process target; this sandbox exposes only one-shot command execution.";
    }
    return "Kimchi ACP supports sandbox remote targets only; this run targets a non-sandbox remote environment.";
  }
  if (!nodeVersionMeetsKimchiAcpMinimum()) {
    return `Node ${process.version} (${process.execPath}) does not satisfy Kimchi ACP's Node >=${MIN_ACP_NODE_VERSION} prerequisite.`;
  }
  const command = resolveKimchiAcpCommand(input.config);
  if (!(await commandIsResolvable(command, resolveConfigPath(input.config), input))) {
    return `Kimchi ACP command is not available: ${command}.`;
  }
  return null;
}

function summarizeStatus(checks: AdapterEnvironmentCheck[]): AdapterEnvironmentTestResult["status"] {
  if (checks.some((check) => check.level === "error")) return "fail";
  if (checks.some((check) => check.level === "warn")) return "warn";
  return "pass";
}

function isNonEmpty(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

export async function testKimchiAcpEnvironment(
  ctx: AdapterEnvironmentTestContext,
): Promise<AdapterEnvironmentTestResult> {
  const checks: AdapterEnvironmentCheck[] = [];
  const config = parseObject(ctx.config);
  const target = ctx.executionTarget ?? null;
  const targetIsRemote = target?.kind === "remote";

  checks.push({
    code: "kimchi_engine_selected",
    level: "info",
    message: "Execution engine selected: ACP.",
    hint: "kimchi_local is ACP-only in v1; there is no CLI JSON-streaming fallback.",
  });

  if (targetIsRemote) {
    checks.push({
      code: "kimchi_acp_remote_target",
      level: "info",
      message: "Kimchi ACP will run against the remote execution environment.",
      hint: "Remote ACP requires a bidirectional process target such as SSH or Paperclip's sandbox process-session bridge.",
    });
  }

  const cwd = asString(config.cwd, process.cwd());
  try {
    await fs.mkdir(cwd, { recursive: true });
    checks.push({
      code: "kimchi_acp_cwd_valid",
      level: "info",
      message: `Working directory is valid: ${cwd}`,
    });
  } catch (err) {
    checks.push({
      code: "kimchi_acp_cwd_invalid",
      level: "error",
      message: err instanceof Error ? err.message : "Invalid working directory",
      detail: cwd,
    });
  }

  checks.push({
    code: nodeVersionMeetsKimchiAcpMinimum() ? "kimchi_acp_node_supported" : "kimchi_acp_node_unsupported",
    level: nodeVersionMeetsKimchiAcpMinimum() ? "info" : "error",
    message: nodeVersionMeetsKimchiAcpMinimum()
      ? `Node ${process.version} satisfies ACP runtime requirements.`
      : `Node ${process.version} (${process.execPath}) does not satisfy ACP runtime requirements.`,
    hint: nodeVersionMeetsKimchiAcpMinimum()
      ? undefined
      : `Run Kimchi ACP with Node >=${MIN_ACP_NODE_VERSION}.`,
  });

  const command = resolveKimchiAcpCommand(config);
  const commandResolvable = await commandIsResolvable(command, resolveConfigPath(config), {
    config,
    executionTarget: ctx.executionTarget,
  });
  checks.push({
    code: commandResolvable ? "kimchi_acp_command_resolvable" : "kimchi_acp_command_missing",
    level: commandResolvable ? "info" : "error",
    message: commandResolvable
      ? `Kimchi ACP command is executable: ${command}`
      : `Kimchi ACP command is not available: ${command}`,
    hint: commandResolvable
      ? undefined
      : "Install the Kimchi CLI (curl -fsSL https://github.com/getkimchi/kimchi/releases/latest/download/install.sh | bash) with ACP support, or set agentCommand to a valid Kimchi ACP server command.",
  });

  // KIMCHI_API_KEY is the env-auth path. Kimchi also supports browser/
  // subscription login inside the CLI; that state lives under
  // ~/.config/kimchi and cannot be probed reliably from here, so a missing
  // key is only a warning.
  const envConfig = parseObject(config.env);
  const apiKeyInConfig = isNonEmpty(envConfig.KIMCHI_API_KEY);
  const apiKeyOnHost = !targetIsRemote && isNonEmpty(process.env.KIMCHI_API_KEY);
  if (apiKeyInConfig || apiKeyOnHost) {
    checks.push({
      code: "kimchi_acp_credentials_detected",
      level: "info",
      message: "Kimchi credentials are set for ACP authentication.",
      detail: `KIMCHI_API_KEY detected in ${apiKeyInConfig ? "adapter config env" : "server environment"}.`,
    });
  } else if (!targetIsRemote) {
    checks.push({
      code: "kimchi_acp_credentials_not_detected",
      level: "warn",
      message: "No Kimchi ACP credentials were detected.",
      hint: "Set KIMCHI_API_KEY in the adapter env, or complete a browser/subscription login inside the Kimchi CLI (`/login`, or the `kimchi setup` wizard) before starting a Kimchi agent.",
    });
  }

  const mode = firstNonEmptyString(config.mode, config.acpMode) ?? DEFAULT_ACP_ENGINE_MODE;
  const warmHandleIdleMs = asNumber(
    config.warmHandleIdleMs ?? config.acpWarmHandleIdleMs,
    DEFAULT_ACP_ENGINE_WARM_HANDLE_IDLE_MS,
  );
  checks.push({
    code: "kimchi_acp_runtime_scaffold",
    level: "info",
    message: "Kimchi ACP runtime execution is available through the shared ACP engine.",
    detail: `mode=${mode}; warmHandleIdleMs=${warmHandleIdleMs}`,
  });

  return {
    adapterType: ctx.adapterType,
    status: summarizeStatus(checks),
    checks,
    testedAt: new Date().toISOString(),
  };
}
