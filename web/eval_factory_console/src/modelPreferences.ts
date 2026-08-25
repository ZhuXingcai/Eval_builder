export type ReasoningEffort =
  | "NONE"
  | "LOW"
  | "MEDIUM"
  | "HIGH"
  | "XHIGH"
  | "MAX";

export type ModelProtocol =
  | "HOST_ROUTED"
  | "OPENAI_RESPONSES"
  | "OPENAI_CHAT_COMPLETIONS"
  | "ANTHROPIC_MESSAGES";

export interface ComposerModelOption {
  id: string;
  displayName: string;
  providerName: string;
  modelId: string;
  protocol: ModelProtocol;
  baseUrl: string | null;
  credentialEnvVar: string | null;
  supportedEfforts: ReasoningEffort[];
  defaultEffort: ReasoningEffort;
  source: "BUILT_IN" | "CUSTOM";
}

export interface CustomModelDraft {
  displayName: string;
  providerName: string;
  modelId: string;
  protocol: Exclude<ModelProtocol, "HOST_ROUTED">;
  baseUrl: string;
  credentialEnvVar: string;
  maximumEffort: ReasoningEffort;
}

export interface ModelSelection {
  modelId: string;
  effort: ReasoningEffort;
}

export const REASONING_EFFORTS: readonly ReasoningEffort[] = [
  "NONE",
  "LOW",
  "MEDIUM",
  "HIGH",
  "XHIGH",
  "MAX",
];

export const EFFORT_LABELS: Readonly<Record<ReasoningEffort, string>> = {
  NONE: "无",
  LOW: "低",
  MEDIUM: "中",
  HIGH: "高",
  XHIGH: "极高",
  MAX: "Max",
};

export const BUILT_IN_MODELS: readonly ComposerModelOption[] = [
  {
    id: "AUTO",
    displayName: "自动模型",
    providerName: "Agent Router",
    modelId: "auto",
    protocol: "HOST_ROUTED",
    baseUrl: null,
    credentialEnvVar: null,
    supportedEfforts: ["LOW", "MEDIUM", "HIGH"],
    defaultEffort: "MEDIUM",
    source: "BUILT_IN",
  },
  {
    id: "CLAUDE",
    displayName: "Claude",
    providerName: "Anthropic",
    modelId: "claude",
    protocol: "ANTHROPIC_MESSAGES",
    baseUrl: null,
    credentialEnvVar: "ANTHROPIC_API_KEY",
    supportedEfforts: [
      "LOW",
      "MEDIUM",
      "HIGH",
      "XHIGH",
      "MAX",
    ],
    defaultEffort: "HIGH",
    source: "BUILT_IN",
  },
  {
    id: "GPT",
    displayName: "GPT",
    providerName: "OpenAI",
    modelId: "gpt",
    protocol: "OPENAI_RESPONSES",
    baseUrl: null,
    credentialEnvVar: "OPENAI_API_KEY",
    supportedEfforts: ["LOW", "MEDIUM", "HIGH", "XHIGH"],
    defaultEffort: "MEDIUM",
    source: "BUILT_IN",
  },
  {
    id: "DEEPSEEK",
    displayName: "DeepSeek",
    providerName: "DeepSeek",
    modelId: "deepseek",
    protocol: "OPENAI_CHAT_COMPLETIONS",
    baseUrl: null,
    credentialEnvVar: "DEEPSEEK_API_KEY",
    supportedEfforts: ["NONE", "LOW", "MEDIUM", "HIGH"],
    defaultEffort: "MEDIUM",
    source: "BUILT_IN",
  },
];

const CUSTOM_MODELS_KEY = "eval-factory.custom-models.v1";
const SESSION_SELECTIONS_KEY = "eval-factory.model-selections.v1";
const MODEL_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:/-]{1,127}$/;
const ENVIRONMENT_NAME_PATTERN = /^[A-Z][A-Z0-9_]{1,63}$/;

export function allComposerModels(
  customModels: readonly ComposerModelOption[],
): readonly ComposerModelOption[] {
  return [...BUILT_IN_MODELS, ...customModels];
}

export function createCustomModel(
  draft: CustomModelDraft,
): ComposerModelOption {
  const displayName = draft.displayName.trim();
  const providerName = draft.providerName.trim();
  const modelId = draft.modelId.trim();
  const baseUrl = normalizeBaseUrl(draft.baseUrl);
  const credentialEnvVar = draft.credentialEnvVar.trim();
  if (displayName.length < 2 || displayName.length > 60) {
    throw new Error("模型名称需为 2 到 60 个字符。");
  }
  if (providerName.length < 2 || providerName.length > 60) {
    throw new Error("服务商名称需为 2 到 60 个字符。");
  }
  if (!MODEL_ID_PATTERN.test(modelId)) {
    throw new Error("模型 ID 只能包含字母、数字、点、横线、下划线、斜杠或冒号。");
  }
  if (
    credentialEnvVar &&
    !ENVIRONMENT_NAME_PATTERN.test(credentialEnvVar)
  ) {
    throw new Error("凭证环境变量需使用大写字母、数字和下划线。");
  }
  if (!isLoopbackUrl(baseUrl) && !credentialEnvVar) {
    throw new Error("远程模型必须提供凭证环境变量名。");
  }
  const supportedEfforts = effortsThrough(draft.maximumEffort);
  return {
    id: `CUSTOM_${safeUuid()}`,
    displayName,
    providerName,
    modelId,
    protocol: draft.protocol,
    baseUrl,
    credentialEnvVar: credentialEnvVar || null,
    supportedEfforts,
    defaultEffort: preferredEffort(supportedEfforts),
    source: "CUSTOM",
  };
}

export function loadCustomModels(): ComposerModelOption[] {
  try {
    const value = JSON.parse(
      window.localStorage.getItem(CUSTOM_MODELS_KEY) ?? "[]",
    );
    if (!Array.isArray(value)) return [];
    return value.flatMap((item) => {
      const model = decodeCustomModel(item);
      return model ? [model] : [];
    });
  } catch {
    return [];
  }
}

export function saveCustomModels(
  models: readonly ComposerModelOption[],
): void {
  try {
    window.localStorage.setItem(
      CUSTOM_MODELS_KEY,
      JSON.stringify(
        models
          .filter((model) => model.source === "CUSTOM")
          .map(encodeCustomModel),
      ),
    );
  } catch {
    // Storage denial leaves the current in-memory configuration usable.
  }
}

export function loadModelSelection(
  sessionId: string,
  models: readonly ComposerModelOption[],
): ModelSelection {
  try {
    const value = JSON.parse(
      window.localStorage.getItem(SESSION_SELECTIONS_KEY) ?? "{}",
    );
    if (isRecord(value) && isModelSelection(value[sessionId])) {
      return normalizeModelSelection(value[sessionId], models);
    }
  } catch {
    // Invalid or unavailable storage falls back to the governed default.
  }
  return defaultModelSelection(models);
}

export function saveModelSelection(
  sessionId: string,
  selection: ModelSelection,
): void {
  try {
    const stored = JSON.parse(
      window.localStorage.getItem(SESSION_SELECTIONS_KEY) ?? "{}",
    );
    const values = isRecord(stored) ? stored : {};
    window.localStorage.setItem(
      SESSION_SELECTIONS_KEY,
      JSON.stringify({ ...values, [sessionId]: selection }),
    );
  } catch {
    // Storage denial leaves the current in-memory preference usable.
  }
}

export function selectionForModel(
  model: ComposerModelOption,
  currentEffort?: ReasoningEffort,
): ModelSelection {
  return {
    modelId: model.id,
    effort:
      currentEffort && model.supportedEfforts.includes(currentEffort)
        ? currentEffort
        : model.defaultEffort,
  };
}

function normalizeModelSelection(
  selection: ModelSelection,
  models: readonly ComposerModelOption[],
): ModelSelection {
  const model = models.find((value) => value.id === selection.modelId);
  if (!model) return defaultModelSelection(models);
  return selectionForModel(model, selection.effort);
}

function defaultModelSelection(
  models: readonly ComposerModelOption[],
): ModelSelection {
  const model = models[0] ?? BUILT_IN_MODELS[0];
  return selectionForModel(model);
}

function effortsThrough(maximum: ReasoningEffort): ReasoningEffort[] {
  const index = REASONING_EFFORTS.indexOf(maximum);
  return REASONING_EFFORTS.slice(0, Math.max(0, index) + 1);
}

function preferredEffort(
  efforts: readonly ReasoningEffort[],
): ReasoningEffort {
  if (efforts.includes("MEDIUM")) return "MEDIUM";
  return efforts.at(-1) ?? "NONE";
}

function normalizeBaseUrl(value: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value.trim());
  } catch {
    throw new Error("请输入有效的模型 API 地址。");
  }
  if (
    !["http:", "https:"].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error("模型 API 地址不能包含凭证、查询参数或片段。");
  }
  if (parsed.protocol !== "https:" && !isLoopbackHost(parsed.hostname)) {
    throw new Error("远程模型必须使用 HTTPS；HTTP 仅限本机地址。");
  }
  return parsed.toString().replace(/\/$/, "");
}

function isLoopbackUrl(value: string): boolean {
  return isLoopbackHost(new URL(value).hostname);
}

function isLoopbackHost(hostname: string): boolean {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "[::1]"
  );
}

function decodeCustomModel(value: unknown): ComposerModelOption | null {
  if (!isRecord(value)) return null;
  const allowedKeys = new Set([
    "id",
    "displayName",
    "providerName",
    "modelId",
    "protocol",
    "baseUrl",
    "credentialEnvVar",
    "supportedEfforts",
    "defaultEffort",
    "source",
  ]);
  if (Object.keys(value).some((key) => !allowedKeys.has(key))) return null;
  const efforts = value.supportedEfforts;
  if (
    typeof value.id === "string" &&
    value.id.startsWith("CUSTOM_") &&
    typeof value.displayName === "string" &&
    value.displayName.length >= 2 &&
    value.displayName.length <= 60 &&
    typeof value.providerName === "string" &&
    value.providerName.length >= 2 &&
    value.providerName.length <= 60 &&
    typeof value.modelId === "string" &&
    MODEL_ID_PATTERN.test(value.modelId) &&
    isModelProtocol(value.protocol) &&
    typeof value.baseUrl === "string" &&
    (value.credentialEnvVar === null ||
      typeof value.credentialEnvVar === "string") &&
    Array.isArray(efforts) &&
    efforts.length > 0 &&
    efforts.every(isReasoningEffort) &&
    new Set(efforts).size === efforts.length &&
    isReasoningEffort(value.defaultEffort) &&
    efforts.includes(value.defaultEffort) &&
    value.source === "CUSTOM"
  ) {
    try {
      if (normalizeBaseUrl(value.baseUrl) !== value.baseUrl) return null;
    } catch {
      return null;
    }
    if (
      value.credentialEnvVar !== null &&
      !ENVIRONMENT_NAME_PATTERN.test(value.credentialEnvVar)
    ) {
      return null;
    }
    if (!isLoopbackUrl(value.baseUrl) && !value.credentialEnvVar) return null;
    return {
      id: value.id,
      displayName: value.displayName,
      providerName: value.providerName,
      modelId: value.modelId,
      protocol: value.protocol,
      baseUrl: value.baseUrl,
      credentialEnvVar: value.credentialEnvVar,
      supportedEfforts: [...efforts],
      defaultEffort: value.defaultEffort,
      source: "CUSTOM",
    };
  }
  return null;
}

function encodeCustomModel(
  model: ComposerModelOption,
): ComposerModelOption {
  return {
    id: model.id,
    displayName: model.displayName,
    providerName: model.providerName,
    modelId: model.modelId,
    protocol: model.protocol,
    baseUrl: model.baseUrl,
    credentialEnvVar: model.credentialEnvVar,
    supportedEfforts: [...model.supportedEfforts],
    defaultEffort: model.defaultEffort,
    source: model.source,
  };
}

function isModelSelection(value: unknown): value is ModelSelection {
  return (
    isRecord(value) &&
    typeof value.modelId === "string" &&
    isReasoningEffort(value.effort)
  );
}

function isModelProtocol(value: unknown): value is ModelProtocol {
  return (
    value === "OPENAI_RESPONSES" ||
    value === "OPENAI_CHAT_COMPLETIONS" ||
    value === "ANTHROPIC_MESSAGES"
  );
}

function isReasoningEffort(value: unknown): value is ReasoningEffort {
  return REASONING_EFFORTS.some((effort) => effort === value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function safeUuid(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  );
}
