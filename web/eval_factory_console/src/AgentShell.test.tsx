import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import AgentShell from "./AgentShell";
import type { AgentShellProjection } from "./agentTypes";

const projection: AgentShellProjection = {
  schema_version: "eval-factory/agent-shell-projection/v1",
  session: {
    schema_version: "eval-factory/agent-shell-session-summary/v1",
    session_id: "session-ui-test",
    title: null,
    status: "ACTIVE",
    session_version: 1,
    last_event_sequence: 1,
    created_at: "2026-08-23T00:00:00Z",
    updated_at: "2026-08-23T00:00:00Z",
  },
  transcript: [],
  activity: [],
  requirement_outcome: null,
  evidence_class: null,
  source_admission_ref: null,
  execution_phase: "WAITING_REQUIREMENT",
  factory: null,
  graph: null,
  team: null,
  plan_reviews: [],
  pending_interactions: [],
  workspaces: [
    {
      schema_version: "eval-factory/agent-shell-workspace/v1",
      kind: "CONVERSATION",
      status: "AVAILABLE",
      owner_refs: [],
      item_count: 0,
      reason_codes: [],
    },
    ...(
      [
        "PLAN_REVIEW",
        "TRACE",
        "TASK",
        "ATTACHMENT",
        "RUBRIC",
        "GRADING",
        "QUALITY",
        "TEAM",
        "ACTIVITY",
        "DELIVERY",
      ] as const
    ).map((kind) => ({
      schema_version:
        "eval-factory/agent-shell-workspace/v1" as const,
      kind,
      status: "EMPTY" as const,
      owner_refs: [],
      item_count: 0,
      reason_codes: [],
    })),
  ],
  delivery: null,
  reconnect_cursor: 1,
  source_fingerprint: "b".repeat(64),
};

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  window.localStorage.clear();
  document.body.replaceChildren();
  window.history.replaceState(null, "", "/");
});

describe("conversation-first Agent Shell", () => {
  it("creates a session and exposes the conversation composer", async () => {
    Object.assign(globalThis, {
      IS_REACT_ACT_ENVIRONMENT: true,
    });
    const fetchMock = vi.fn(
      async (
        input: RequestInfo | URL,
        init?: RequestInit,
      ): Promise<Response> => {
        const url = String(input);
        if (url.endsWith("/api/harness/contract")) {
          return json({
            schema_version:
              "eval-factory/agent-shell-api-contract/v1",
            principal_header: "X-Eval-Factory-Principal",
            event_stream_media_type: "text/event-stream",
            event_families: [],
            max_event_page_size: 500,
          });
        }
        if (url.endsWith("/api/harness/source-admission/contract")) {
          return json({
            schema_version:
              "eval-factory/agent-shell-source-admission-contract/v1",
            manifest_name: "manifest.csv",
            manifest_columns: [],
            max_source_files: 500,
            max_manifest_bytes: 1024,
            max_source_bytes: 1024,
            max_total_source_bytes: 4096,
            max_request_bytes: 8192,
            error_codes: [],
          });
        }
        if (url.includes("/api/harness/sessions?")) {
          return json({
            schema_version:
              "eval-factory/agent-shell-session-page/v1",
            offset: 0,
            limit: 100,
            total: 0,
            sessions: [],
          });
        }
        if (
          url.endsWith("/api/harness/sessions") &&
          init?.method === "POST"
        ) {
          return json(projection);
        }
        if (
          url.endsWith("/api/harness/sessions/session-ui-test") &&
          init?.method === "DELETE"
        ) {
          return json({
            ...projection,
            session: {
              ...projection.session,
              status: "CLOSED",
              session_version: 2,
              last_event_sequence: 2,
            },
          });
        }
        if (url.includes("/stream?")) {
          return new Response("", {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
          });
        }
        throw new Error(`Unexpected request: ${url}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<AgentShell />);
      await settle();
    });

    expect(container.textContent).toContain("暂无会话");
    const create = Array.from(
      container.querySelectorAll<HTMLButtonElement>("button"),
    ).find((button) => button.textContent?.includes("新建会话"));
    expect(create).not.toBeUndefined();

    await act(async () => {
      create?.click();
      await settle();
    });

    expect(container.textContent).toContain("等待评测需求");
    expect(
      container.querySelector<HTMLTextAreaElement>(
        'textarea[placeholder="描述评测目标、数据边界和交付要求…"]',
      ),
    ).not.toBeNull();
    expect(
      Array.from(
        container.querySelectorAll<HTMLButtonElement>(
          ".workspace-navigation > button",
        ),
      ).map((button) => button.textContent?.replace(/\s+/g, " ").trim()),
    ).toEqual([
      "对话目标与上下文",
      "评测工作台来源、准则与交付",
      "运行记录Team、Graph 与活动",
    ]);
    expect(
      container.querySelector(".agent-main")?.classList,
    ).toContain("composer-centered");
    expect(
      container.querySelector<HTMLSelectElement>(
        'select[title="按 Balanced Autonomy 继续执行"]',
      )?.value,
    ).toBe("AGENT");
    const addContext =
      container.querySelector<HTMLButtonElement>(
        'button[aria-label="添加上下文"]',
      );
    expect(addContext).not.toBeNull();
    expect(
      container.querySelector('select[aria-label="模型偏好"]'),
    ).toBeNull();
    expect(
      container.querySelector('select[aria-label="思考深度"]'),
    ).toBeNull();
    const modelPicker =
      container.querySelector<HTMLButtonElement>(
        'button[aria-label="选择模型和思考深度"]',
      );
    expect(modelPicker?.textContent).toContain("自动模型");
    expect(modelPicker?.textContent).toContain("中");
    await act(async () => {
      modelPicker?.click();
    });
    expect(
      container.querySelector(
        '[role="dialog"][aria-label="模型和思考深度"]',
      ),
    ).not.toBeNull();
    const claude = Array.from(
      container.querySelectorAll<HTMLButtonElement>(".model-option"),
    ).find((button) => button.textContent?.includes("Claude"));
    await act(async () => {
      claude?.click();
    });
    const effort = container.querySelector<HTMLInputElement>(
      'input[aria-label="Claude 思考深度"]',
    );
    expect(effort?.value).toBe("2");
    await act(async () => {
      setInputValue(effort, "4");
    });
    expect(modelPicker?.textContent).toContain("Claude");
    expect(modelPicker?.textContent).toContain("Max");

    const addCustom = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        ".model-picker-footer button",
      ),
    ).find((button) => button.textContent?.includes("自定义模型"));
    await act(async () => {
      addCustom?.click();
    });
    const customForm =
      container.querySelector<HTMLFormElement>(".custom-model-form");
    await act(async () => {
      setInputValue(
        customForm?.querySelector<HTMLInputElement>(
          'input[placeholder="例如：团队代码模型"]',
        ) ?? null,
        "团队模型",
      );
      setInputValue(
        customForm?.querySelector<HTMLInputElement>(
          'input[placeholder="例如：OpenRouter"]',
        ) ?? null,
        "OpenRouter",
      );
      setInputValue(
        customForm?.querySelector<HTMLInputElement>(
          'input[placeholder="provider/model-name"]',
        ) ?? null,
        "vendor/reasoner",
      );
      setInputValue(
        customForm?.querySelector<HTMLInputElement>(
          'input[placeholder="https://gateway.example.com/v1"]',
        ) ?? null,
        "https://gateway.example.com/v1",
      );
      setInputValue(
        customForm?.querySelector<HTMLInputElement>(
          'input[placeholder="CUSTOM_API_KEY"]',
        ) ?? null,
        "TEAM_MODEL_API_KEY",
      );
      customForm?.dispatchEvent(
        new Event("submit", {
          bubbles: true,
          cancelable: true,
        }),
      );
    });
    expect(modelPicker?.textContent).toContain("团队模型");
    expect(window.localStorage.getItem(
      "eval-factory.custom-models.v1",
    )).toContain("vendor/reasoner");
    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          'button[aria-label="关闭模型选择"]',
        )
        ?.click();
    });
    expect(
      container.querySelector<HTMLButtonElement>(
        'button[aria-label="语音输入"]',
      ),
    ).not.toBeNull();
    expect(
      container.querySelector(".composer-toolbar .send-button"),
    ).not.toBeNull();
    await act(async () => {
      container
        .querySelector<HTMLButtonElement>(
          'button[aria-label="语音输入"]',
        )
        ?.click();
    });
    expect(container.textContent).toContain(
      "当前浏览器不支持语音输入",
    );

    await act(async () => {
      addContext?.click();
    });
    expect(
      Array.from(
        container.querySelectorAll<HTMLButtonElement>(
          ".composer-add-menu > .composer-menu-item",
        ),
      ).map((button) =>
        button.querySelector("strong")?.textContent?.trim(),
      ),
    ).toEqual([
      "文件和文件夹",
      "在项目中使用",
      "目标",
      "计划模式",
      "插件",
    ]);
    expect(container.textContent).toContain("需要桌面 Host");
    const sources = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        ".composer-add-menu > .composer-menu-item",
      ),
    ).find((button) =>
      button.textContent?.includes("文件和文件夹"),
    );
    await act(async () => {
      sources?.click();
    });
    expect(container.textContent).toContain("选择文件夹");

    const closeContext =
      container.querySelector<HTMLButtonElement>(
        'button[aria-label="关闭添加上下文"]',
      );
    await act(async () => {
      closeContext?.click();
    });

    const textarea =
      container.querySelector<HTMLTextAreaElement>(
        'textarea[name="evaluation-requirement"]',
      );
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        "value",
      )?.set;
      setter?.call(textarea, "/goal");
      textarea?.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(container.querySelector(".composer-command-menu")).not.toBeNull();
    const goalCommand = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        ".composer-command-menu button",
      ),
    ).find((button) => button.textContent?.includes("/goal"));
    await act(async () => {
      goalCommand?.click();
    });
    expect(textarea?.value).toContain("验收标准");

    const mode = container.querySelector<HTMLSelectElement>(
      'select[title="按 Balanced Autonomy 继续执行"]',
    );
    await act(async () => {
      if (!mode) return;
      mode.value = "PLAN";
      mode.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(
      container.querySelector<HTMLSelectElement>(
        'select[title="提交后优先进入计划审核"]',
      )?.value,
    ).toBe("PLAN");
    expect(window.location.pathname).toBe(
      "/sessions/session-ui-test",
    );
    expect(
      fetchMock.mock.calls.some(
        ([, init]) =>
          (init as RequestInit | undefined)?.headers &&
          (
            (init as RequestInit).headers as Record<string, string>
          )["X-Eval-Factory-Principal"] === "user://agent-shell",
      ),
    ).toBe(true);

    const deleteSession =
      container.querySelector<HTMLButtonElement>(
        'button[aria-label="删除会话 新会话"]',
      );
    await act(async () => {
      deleteSession?.click();
    });
    expect(
      container.querySelector("#close-session-dialog"),
    ).not.toBeNull();
    const confirmDelete = Array.from(
      container.querySelectorAll<HTMLButtonElement>(
        "#close-session-dialog button",
      ),
    ).find((button) => button.textContent?.includes("删除会话"));
    await act(async () => {
      confirmDelete?.click();
      await settle();
    });
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) =>
          String(url).endsWith(
            "/api/harness/sessions/session-ui-test",
          ) && init?.method === "DELETE",
      ),
    ).toBe(true);
    expect(container.textContent).toContain("暂无会话");
    expect(window.location.pathname).toBe("/");

    await act(async () => root.unmount());
  });

  it("falls back to the compatible PlanReview workspace", async () => {
    Object.assign(globalThis, {
      IS_REACT_ACT_ENVIRONMENT: true,
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/harness/contract")) {
          return json(
            { detail: "Not Found" },
            404,
          );
        }
        if (url.endsWith("/api/plan-reviews/contract")) {
          return json({
            schema_version:
              "eval-factory/plan-review-api-contract/v1",
            decision_actions: [],
            review_states: [],
            principal_header: "X-Eval-Factory-Principal",
          });
        }
        return json({
          schema_version: "eval-factory/plan-review-page/v1",
          items: [],
          offset: 0,
          limit: 100,
          total: 0,
        });
      }),
    );
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(<AgentShell />);
      await settle();
      await settle();
    });

    expect(container.textContent).toContain("审核队列");
    expect(container.textContent).toContain("暂无待审核计划");

    await act(async () => root.unmount());
  });
});

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function setInputValue(
  input: HTMLInputElement | null,
  value: string,
): void {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype,
    "value",
  )?.set;
  setter?.call(input, value);
  input?.dispatchEvent(new Event("input", { bubbles: true }));
}

async function settle() {
  await new Promise((resolve) => window.setTimeout(resolve, 0));
}
