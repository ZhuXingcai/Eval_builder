import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  BUILT_IN_MODELS,
  allComposerModels,
  createCustomModel,
  loadCustomModels,
  loadModelSelection,
  saveCustomModels,
  saveModelSelection,
  selectionForModel,
  type ComposerModelOption,
} from "./modelPreferences";

beforeEach(() => {
  window.localStorage.clear();
  vi.stubGlobal("crypto", {
    randomUUID: () => "00000000-0000-4000-8000-000000000003",
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("model preferences", () => {
  it("creates a custom model without accepting a plaintext credential", () => {
    const model = createCustomModel({
      displayName: "Team Reasoner",
      providerName: "OpenRouter",
      modelId: "vendor/reasoner-v1",
      protocol: "OPENAI_RESPONSES",
      baseUrl: "https://gateway.example.com/v1/",
      credentialEnvVar: "TEAM_MODEL_API_KEY",
      maximumEffort: "XHIGH",
    });

    expect(model).toEqual({
      id: "CUSTOM_00000000-0000-4000-8000-000000000003",
      displayName: "Team Reasoner",
      providerName: "OpenRouter",
      modelId: "vendor/reasoner-v1",
      protocol: "OPENAI_RESPONSES",
      baseUrl: "https://gateway.example.com/v1",
      credentialEnvVar: "TEAM_MODEL_API_KEY",
      supportedEfforts: [
        "NONE",
        "LOW",
        "MEDIUM",
        "HIGH",
        "XHIGH",
      ],
      defaultEffort: "MEDIUM",
      source: "CUSTOM",
    });
  });

  it("rejects unsafe endpoints and missing remote credential references", () => {
    expect(() =>
      createCustomModel({
        displayName: "Unsafe Model",
        providerName: "Provider",
        modelId: "unsafe-model",
        protocol: "OPENAI_CHAT_COMPLETIONS",
        baseUrl: "https://user:secret@example.com/v1",
        credentialEnvVar: "MODEL_API_KEY",
        maximumEffort: "HIGH",
      }),
    ).toThrow("不能包含凭证");
    expect(() =>
      createCustomModel({
        displayName: "Remote Model",
        providerName: "Provider",
        modelId: "remote-model",
        protocol: "OPENAI_RESPONSES",
        baseUrl: "https://gateway.example.com/v1",
        credentialEnvVar: "",
        maximumEffort: "HIGH",
      }),
    ).toThrow("必须提供凭证环境变量名");
  });

  it("allows a credential-free loopback model", () => {
    const model = createCustomModel({
      displayName: "Local Model",
      providerName: "Ollama",
      modelId: "qwen3-coder",
      protocol: "OPENAI_CHAT_COMPLETIONS",
      baseUrl: "http://127.0.0.1:11434/v1",
      credentialEnvVar: "",
      maximumEffort: "NONE",
    });

    expect(model.credentialEnvVar).toBeNull();
    expect(model.supportedEfforts).toEqual(["NONE"]);
  });

  it("persists only strict custom model metadata", () => {
    const model = createCustomModel({
      displayName: "Stored Model",
      providerName: "Gateway",
      modelId: "team/model",
      protocol: "ANTHROPIC_MESSAGES",
      baseUrl: "https://models.example.com",
      credentialEnvVar: "GATEWAY_TOKEN",
      maximumEffort: "MAX",
    });
    const taintedModel = {
      ...model,
      apiKey: "must-not-survive",
    } as ComposerModelOption;
    saveCustomModels([
      BUILT_IN_MODELS[0],
      taintedModel,
    ]);
    expect(
      window.localStorage.getItem("eval-factory.custom-models.v1"),
    ).not.toContain("must-not-survive");
    expect(loadCustomModels()).toEqual([model]);

    window.localStorage.setItem(
      "eval-factory.custom-models.v1",
      JSON.stringify([{ ...model, apiKey: "must-not-survive" }]),
    );
    expect(loadCustomModels()).toEqual([]);
  });

  it("rejects a stored remote model without a credential reference", () => {
    const model = createCustomModel({
      displayName: "Stored Remote",
      providerName: "Gateway",
      modelId: "team/model",
      protocol: "OPENAI_RESPONSES",
      baseUrl: "https://models.example.com",
      credentialEnvVar: "GATEWAY_TOKEN",
      maximumEffort: "HIGH",
    });
    window.localStorage.setItem(
      "eval-factory.custom-models.v1",
      JSON.stringify([{ ...model, credentialEnvVar: null }]),
    );

    expect(loadCustomModels()).toEqual([]);
  });

  it("keeps model and effort together per session", () => {
    const models = allComposerModels([]);
    const selection = selectionForModel(models[1], "XHIGH");
    saveModelSelection("session-one", selection);

    expect(loadModelSelection("session-one", models)).toEqual(selection);
    expect(loadModelSelection("session-two", models)).toEqual({
      modelId: "AUTO",
      effort: "MEDIUM",
    });
  });

  it("clamps an unsupported stored effort to the model default", () => {
    const models = allComposerModels([]);
    saveModelSelection("session-one", {
      modelId: "DEEPSEEK",
      effort: "MAX",
    });

    expect(loadModelSelection("session-one", models)).toEqual({
      modelId: "DEEPSEEK",
      effort: "MEDIUM",
    });
  });
});
