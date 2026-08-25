import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type CSSProperties,
  type KeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
  type SyntheticEvent,
} from "react";
import { createPortal } from "react-dom";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowUp,
  Bot,
  Boxes,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  Database,
  FileJson2,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  GitBranch,
  ListChecks,
  Menu,
  MessageSquare,
  Mic,
  PackageCheck,
  Paperclip,
  Plug,
  Plus,
  Quote,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
  Target,
  Trash2,
  UploadCloud,
  User,
  Users,
  Wifi,
  WifiOff,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react";

import PlanReviewWorkspace from "./App";
import {
  admitAgentSource,
  closeAgentSession,
  createAgentSession,
  fetchAgentShellContract,
  fetchSourceContract,
  getAgentMember,
  getAgentSession,
  getAgentSource,
  listAgentSessions,
  postAgentMessage,
  reconcileAgentSession,
  subscribeAgentSession,
} from "./agentApi";
import type {
  AgentShellConnectionState,
  AgentShellEvent,
  AgentShellExecutionPhase,
  AgentShellInteractionCard,
  AgentShellMemberProjection,
  AgentShellProjection,
  AgentShellSessionSummary,
  AgentShellSourceAdmission,
  AgentShellSourceContract,
  AgentShellTeamMemberSummary,
  AgentShellWorkspaceDescriptor,
  AgentShellWorkspaceKind,
} from "./agentTypes";
import { ApiError } from "./api";
import {
  BUILT_IN_MODELS,
  EFFORT_LABELS,
  allComposerModels,
  createCustomModel,
  loadCustomModels,
  loadModelSelection,
  saveCustomModels,
  saveModelSelection,
  selectionForModel,
  type ComposerModelOption,
  type CustomModelDraft,
  type ModelSelection,
  type ReasoningEffort,
} from "./modelPreferences";
import type { PlanReviewView } from "./types";

const PRINCIPAL = "user://agent-shell";

type InspectorView = "activity" | "team" | null;
type HostMode = "loading" | "shell" | "plan-review";
type ShellMode = "CONVERSATION" | "WORKBENCH" | "RUN";
type ComposerMode = "AGENT" | "PLAN";
type ComposerPanel = "ADD" | "EXTENSIONS" | "MODEL" | null;

interface ComposerQuote {
  messageId: string;
  content: string;
}

interface ComposerCommand {
  command: string;
  label: string;
  description: string;
  icon: LucideIcon;
}

interface BrowserSpeechRecognition {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onresult:
    | ((event: {
        resultIndex: number;
        results: ArrayLike<{
          0: { transcript: string };
          isFinal: boolean;
        }>;
      }) => void)
    | null;
  start: () => void;
  stop: () => void;
}

type BrowserSpeechRecognitionConstructor =
  new () => BrowserSpeechRecognition;

const WORKSPACE_LABELS: Record<AgentShellWorkspaceKind, string> = {
  CONVERSATION: "对话",
  PLAN_REVIEW: "计划审核",
  TRACE: "轨迹",
  TASK: "任务",
  ATTACHMENT: "附件",
  RUBRIC: "准则",
  GRADING: "评分",
  QUALITY: "质量",
  TEAM: "团队",
  ACTIVITY: "动态",
  DELIVERY: "交付",
};

const WORKSPACE_ICONS: Record<
  AgentShellWorkspaceKind,
  LucideIcon
> = {
  CONVERSATION: MessageSquare,
  PLAN_REVIEW: ListChecks,
  TRACE: GitBranch,
  TASK: Workflow,
  ATTACHMENT: Paperclip,
  RUBRIC: FileText,
  GRADING: FileJson2,
  QUALITY: ShieldCheck,
  TEAM: Users,
  ACTIVITY: Activity,
  DELIVERY: PackageCheck,
};

const WORKBENCH_WORKSPACES = [
  "TRACE",
  "TASK",
  "ATTACHMENT",
  "RUBRIC",
  "GRADING",
  "QUALITY",
  "DELIVERY",
  "PLAN_REVIEW",
] as const satisfies readonly AgentShellWorkspaceKind[];

const RUN_WORKSPACES = [
  "TEAM",
  "ACTIVITY",
] as const satisfies readonly AgentShellWorkspaceKind[];

const SHELL_MODES: ReadonlyArray<{
  mode: ShellMode;
  label: string;
  compactLabel: string;
  icon: LucideIcon;
}> = [
  {
    mode: "CONVERSATION",
    label: "对话",
    compactLabel: "对话",
    icon: MessageSquare,
  },
  {
    mode: "WORKBENCH",
    label: "评测工作台",
    compactLabel: "工作台",
    icon: Boxes,
  },
  {
    mode: "RUN",
    label: "运行记录",
    compactLabel: "运行",
    icon: Activity,
  },
];

const COMPOSER_COMMANDS: readonly ComposerCommand[] = [
  {
    command: "/goal",
    label: "设定目标",
    description: "插入目标、验收标准和约束模板",
    icon: Target,
  },
  {
    command: "/plan",
    label: "计划模式",
    description: "提交后优先进入计划审核",
    icon: ListChecks,
  },
  {
    command: "/agent",
    label: "Agent 模式",
    description: "按 Balanced Autonomy 继续执行",
    icon: Bot,
  },
  {
    command: "/model",
    label: "模型与思考",
    description: "选择模型并调整该模型支持的思考深度",
    icon: Settings2,
  },
  {
    command: "/source",
    label: "添加来源",
    description: "选择 manifest.csv、JSONL 或来源目录",
    icon: Paperclip,
  },
  {
    command: "/skills",
    label: "Skills 与插件",
    description: "查看当前可用能力与扩展状态",
    icon: Sparkles,
  },
];

const GOAL_TEMPLATE = `目标：

验收标准：
-

约束：
-`;

export default function AgentShell() {
  const [hostMode, setHostMode] = useState<HostMode>("loading");
  const [sessions, setSessions] = useState<AgentShellSessionSummary[]>([]);
  const [current, setCurrent] = useState<AgentShellProjection | null>(null);
  const [source, setSource] = useState<AgentShellSourceAdmission | null>(null);
  const [sourceContract, setSourceContract] =
    useState<AgentShellSourceContract | null>(null);
  const [workspace, setWorkspace] =
    useState<AgentShellWorkspaceKind>("CONVERSATION");
  const [inspector, setInspector] = useState<InspectorView>(null);
  const [member, setMember] =
    useState<AgentShellMemberProjection | null>(null);
  const [navigationOpen, setNavigationOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [draft, setDraft] = useState("");
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [composerMode, setComposerMode] =
    useState<ComposerMode>("AGENT");
  const [customModels, setCustomModels] = useState<
    ComposerModelOption[]
  >(loadCustomModels);
  const composerModels = useMemo(
    () => allComposerModels(customModels),
    [customModels],
  );
  const [modelSelection, setModelSelection] =
    useState<ModelSelection>(() =>
      selectionForModel(BUILT_IN_MODELS[0]),
    );
  const [composerQuotes, setComposerQuotes] = useState<ComposerQuote[]>(
    [],
  );
  const [sessionToClose, setSessionToClose] =
    useState<AgentShellSessionSummary | null>(null);
  const [operation, setOperation] = useState<string | null>(null);
  const [notice, setNotice] = useState("正在连接 Agent Host");
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] =
    useState<AgentShellConnectionState>("connecting");
  const refreshTimer = useRef<number | null>(null);
  const navigationModal = useMediaQuery("(max-width: 900px)");
  const inspectorModal = useMediaQuery("(max-width: 1199px)");

  useEffect(() => {
    const sessionId = current?.session.session_id;
    if (!sessionId) return;
    setModelSelection(
      loadModelSelection(sessionId, composerModels),
    );
  }, [composerModels, current?.session.session_id]);

  useOverlayFocus(
    navigationOpen,
    "#session-navigation",
    navigationModal,
    () => setNavigationOpen(false),
  );
  useOverlayFocus(
    inspector !== null,
    "#agent-inspector",
    inspectorModal,
    () => setInspector(null),
  );
  useOverlayFocus(
    sessionToClose !== null,
    "#close-session-dialog",
    true,
    dismissSessionClose,
  );

  const loadSession = useCallback(
    async (
      sessionId: string,
      options: { updateRoute?: boolean; nextWorkspace?: AgentShellWorkspaceKind } = {},
    ) => {
      const [projection, admitted] = await Promise.all([
        getAgentSession(sessionId),
        getAgentSource(sessionId),
      ]);
      setCurrent(projection);
      setSource(admitted);
      setSessions((values) => upsertSession(values, projection.session));
      const nextWorkspace = options.nextWorkspace ?? workspaceFromLocation().workspace;
      setWorkspace(nextWorkspace);
      setMember(null);
      setNavigationOpen(false);
      setError(null);
      if (options.updateRoute !== false) {
        writeRoute(sessionId, nextWorkspace);
      }
      return projection;
    },
    [],
  );

  const refreshSessions = useCallback(async () => {
    const page = await listAgentSessions();
    setSessions(page.sessions);
    return page.sessions;
  }, []);

  useEffect(() => {
    let active = true;
    const bootstrap = async () => {
      try {
        await fetchAgentShellContract();
      } catch (reason) {
        if (reason instanceof ApiError && reason.status === 404) {
          if (active) setHostMode("plan-review");
          return;
        }
        if (active) {
          setHostMode("shell");
          setConnection("offline");
          setError(errorMessage(reason));
          setNotice("Agent Host 不可用");
        }
        return;
      }
      if (!active) return;
      setHostMode("shell");
      try {
        const [available, contract] = await Promise.all([
          refreshSessions(),
          fetchSourceContract(),
        ]);
        if (!active) return;
        setSourceContract(contract);
        const route = workspaceFromLocation();
        const selected =
          available.find(
            (item) => item.session_id === route.sessionId,
          ) ??
          available[0] ??
          null;
        if (selected) {
          await loadSession(selected.session_id, {
            updateRoute: true,
            nextWorkspace: route.workspace,
          });
        } else {
          setNotice("尚无会话");
          setConnection("online");
        }
      } catch (reason) {
        if (!active) return;
        setConnection("offline");
        setError(errorMessage(reason));
        setNotice("Agent Host 不可用");
      }
    };
    void bootstrap();
    return () => {
      active = false;
    };
  }, [loadSession, refreshSessions]);

  useEffect(() => {
    const onPopState = () => {
      const route = workspaceFromLocation();
      if (!route.sessionId) {
        setCurrent(null);
        setWorkspace("CONVERSATION");
        return;
      }
      void loadSession(route.sessionId, {
        updateRoute: false,
        nextWorkspace: route.workspace,
      }).catch((reason) => setError(errorMessage(reason)));
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [loadSession]);

  useEffect(() => {
    const sessionId = current?.session.session_id;
    if (!sessionId) {
      setConnection(hostMode === "shell" ? "online" : "connecting");
      return;
    }
    return subscribeAgentSession(
      sessionId,
      current.reconnect_cursor,
      () => {
        if (refreshTimer.current !== null) return;
        refreshTimer.current = window.setTimeout(() => {
          refreshTimer.current = null;
          void loadSession(sessionId, {
            updateRoute: false,
          }).catch((reason) => {
            setError(errorMessage(reason));
            setConnection("offline");
          });
        }, 100);
      },
      setConnection,
    );
  }, [current?.session.session_id, hostMode, loadSession]);

  useEffect(
    () => () => {
      if (refreshTimer.current !== null) {
        window.clearTimeout(refreshTimer.current);
      }
    },
    [],
  );

  const filteredSessions = useMemo(() => {
    const value = query.trim().toLocaleLowerCase();
    if (!value) return sessions;
    return sessions.filter((session) =>
      `${session.title ?? ""} ${session.session_id} ${session.status}`
        .toLocaleLowerCase()
        .includes(value),
    );
  }, [query, sessions]);

  const busy = operation !== null;

  async function runOperation(
    label: string,
    action: () => Promise<void>,
    success: string,
  ) {
    setOperation(label);
    setError(null);
    try {
      await action();
      setNotice(success);
    } catch (reason) {
      setError(errorMessage(reason));
      setNotice("操作未提交");
    } finally {
      setOperation(null);
    }
  }

  function createSession() {
    void runOperation(
      "正在创建会话",
      async () => {
        const projection = await createAgentSession(PRINCIPAL);
        setCurrent(projection);
        setSource(null);
        setSessions((values) => upsertSession(values, projection.session));
        setWorkspace("CONVERSATION");
        setNavigationOpen(false);
        writeRoute(projection.session.session_id, "CONVERSATION");
      },
      "新会话已创建",
    );
  }

  function chooseSession(sessionId: string) {
    void runOperation(
      "正在打开会话",
      async () => {
        await loadSession(sessionId, {
          updateRoute: true,
          nextWorkspace: "CONVERSATION",
        });
      },
      "会话已同步",
    );
  }

  function dismissSessionClose() {
    const sessionId = sessionToClose?.session_id;
    setSessionToClose(null);
    if (!sessionId) return;
    window.setTimeout(() => {
      const trigger = Array.from(
        document.querySelectorAll<HTMLButtonElement>(
          ".session-delete-button",
        ),
      ).find((button) => button.dataset.sessionId === sessionId);
      trigger?.focus();
    }, 0);
  }

  function closeSession() {
    if (!sessionToClose) return;
    const closing = sessionToClose;
    void runOperation(
      "正在删除会话",
      async () => {
        await closeAgentSession(closing, PRINCIPAL);
        const remaining = sessions.filter(
          (session) => session.session_id !== closing.session_id,
        );
        setSessions(remaining);
        setSessionToClose(null);
        if (current?.session.session_id !== closing.session_id) return;
        setCurrent(null);
        setSource(null);
        setMember(null);
        setInspector(null);
        setPendingFiles([]);
        setComposerQuotes([]);
        setDraft("");
        setWorkspace("CONVERSATION");
        const next = remaining[0];
        if (next) {
          await loadSession(next.session_id, {
            updateRoute: true,
            nextWorkspace: "CONVERSATION",
          });
        } else {
          window.history.pushState(null, "", "/");
        }
      },
      "会话已从列表移除",
    );
  }

  function chooseWorkspace(kind: AgentShellWorkspaceKind) {
    if (!current) return;
    setWorkspace(kind);
    setMember(null);
    setInspector(null);
    writeRoute(current.session.session_id, kind);
  }

  function chooseShellMode(mode: ShellMode) {
    if (!current) return;
    chooseWorkspace(resolveModeWorkspace(mode, current, workspace));
  }

  function selectFiles(event: ChangeEvent<HTMLInputElement>) {
    setPendingFiles(uniqueSourceFiles(event.target.files));
    setError(null);
    event.target.value = "";
  }

  function selectDirectory(event: ChangeEvent<HTMLInputElement>) {
    const selected = uniqueSourceFiles(event.target.files).filter(
      (file) =>
        file.name === "manifest.csv" ||
        file.name.toLocaleLowerCase().endsWith(".jsonl"),
    );
    if (!selected.length) {
      setError("所选目录不包含 manifest.csv 或 JSONL 来源。");
      event.target.value = "";
      return;
    }
    setPendingFiles(selected);
    setError(null);
    setNotice(`已从目录选择 ${selected.length} 个来源文件`);
    event.target.value = "";
  }

  function insertGoalTemplate() {
    setDraft((value) =>
      value.trim() ? `${value.trimEnd()}\n\n${GOAL_TEMPLATE}` : GOAL_TEMPLATE,
    );
  }

  function chooseModel(selection: ModelSelection) {
    setModelSelection(selection);
    if (current) {
      saveModelSelection(current.session.session_id, selection);
    }
  }

  function addCustomModel(draft: CustomModelDraft): string | null {
    try {
      const model = createCustomModel(draft);
      if (
        composerModels.some(
          (value) =>
            value.providerName.toLocaleLowerCase() ===
              model.providerName.toLocaleLowerCase() &&
            value.modelId.toLocaleLowerCase() ===
              model.modelId.toLocaleLowerCase() &&
            value.baseUrl === model.baseUrl,
        )
      ) {
        return "这个服务商、地址和模型 ID 已经存在。";
      }
      const next = [...customModels, model];
      const selection = selectionForModel(model);
      saveCustomModels(next);
      setCustomModels(next);
      chooseModel(selection);
      return null;
    } catch (reason) {
      return errorMessage(reason);
    }
  }

  function deleteCustomModel(modelId: string) {
    const next = customModels.filter((model) => model.id !== modelId);
    saveCustomModels(next);
    setCustomModels(next);
    if (modelSelection.modelId !== modelId) return;
    chooseModel(selectionForModel(BUILT_IN_MODELS[0]));
  }

  function quoteAgentOutput(messageId: string, content: string) {
    const bounded = content.trim().slice(0, 1_200);
    if (!bounded) return;
    setComposerQuotes((values) => {
      if (
        values.some(
          (quote) =>
            quote.messageId === messageId && quote.content === bounded,
        )
      ) {
        return values;
      }
      return [
        ...values.slice(-4),
        { messageId, content: bounded },
      ];
    });
    setWorkspace("CONVERSATION");
    if (current) writeRoute(current.session.session_id, "CONVERSATION");
    window.requestAnimationFrame(() => {
      document
        .querySelector<HTMLTextAreaElement>(
          'textarea[name="evaluation-requirement"]',
        )
        ?.focus();
    });
  }

  function admitSource() {
    if (!current || !pendingFiles.length) return;
    void runOperation(
      "正在校验并接入来源",
      async () => {
        const admitted = await admitAgentSource(
          current,
          pendingFiles,
          PRINCIPAL,
        );
        setSource(admitted);
        setPendingFiles([]);
        await loadSession(current.session.session_id, {
          updateRoute: false,
          nextWorkspace: workspace,
        });
      },
      "来源已接入",
    );
  }

  function sendMessage() {
    const content = composeMessageContent(draft, composerQuotes);
    if (!current || !content) return;
    if (pendingFiles.length) {
      setError("请先确认接入已选择的来源文件。");
      return;
    }
    if (content.length > 32_768) {
      setError("需求与引用内容合计不能超过 32,768 个字符。");
      return;
    }
    void runOperation(
      "Agent 正在处理",
      async () => {
        const projection = await postAgentMessage(
          current,
          content,
          PRINCIPAL,
          source,
        );
        setCurrent(projection);
        setSessions((values) => upsertSession(values, projection.session));
        setDraft("");
        setComposerQuotes([]);
        if (
          composerMode === "PLAN" &&
          projection.plan_reviews.length
        ) {
          setWorkspace("PLAN_REVIEW");
          writeRoute(
            projection.session.session_id,
            "PLAN_REVIEW",
          );
        }
      },
      "已提交到 Harness",
    );
  }

  function reconcile() {
    if (!current) return;
    void runOperation(
      "正在同步执行",
      async () => {
        const projection = await reconcileAgentSession(
          current,
          PRINCIPAL,
        );
        setCurrent(projection);
        setSessions((values) => upsertSession(values, projection.session));
      },
      "执行状态已同步",
    );
  }

  function openMember(value: AgentShellTeamMemberSummary) {
    if (!current) return;
    setInspector("team");
    setNavigationOpen(false);
    writeMemberRoute(current.session.session_id, value.member_id);
    void getAgentMember(
      current.session.session_id,
      value.member_id,
    )
      .then(setMember)
      .catch((reason) => setError(errorMessage(reason)));
  }

  function onComposerKeyDown(
    event: KeyboardEvent<HTMLTextAreaElement>,
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      sendMessage();
    }
  }

  function onReviewChange(view: PlanReviewView) {
    if (view.result.state === "RESUMED") {
      reconcile();
      return;
    }
    if (current) {
      void loadSession(current.session.session_id, {
        updateRoute: false,
        nextWorkspace: "PLAN_REVIEW",
      });
    }
  }

  if (hostMode === "plan-review") {
    return <PlanReviewWorkspace />;
  }

  if (hostMode === "loading") {
    return (
      <div className="shell-boot" role="status">
        <span className="shell-spinner" aria-hidden="true" />
        <strong>正在连接评测 Agent</strong>
      </div>
    );
  }

  const composerCentered =
    workspace === "CONVERSATION" &&
    !!current &&
    !current.transcript.length &&
    !current.pending_interactions.length;

  return (
    <div
      className={[
        "agent-shell",
        inspector ? "inspector-open" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <a className="skip-link" href="#agent-workspace">
        跳转到会话
      </a>
      {(navigationOpen || inspector) && (
        <button
          type="button"
          className="shell-backdrop"
          aria-label="关闭面板"
          onClick={() => {
            setNavigationOpen(false);
            setInspector(null);
          }}
        />
      )}
      <SessionRail
        currentId={current?.session.session_id ?? null}
        modal={navigationOpen && navigationModal}
        mobileOpen={navigationOpen}
        onClose={() => setNavigationOpen(false)}
        onCreate={createSession}
        onDelete={setSessionToClose}
        onOpen={chooseSession}
        onQuery={setQuery}
        query={query}
        sessions={filteredSessions}
        workspace={workspace}
        workspaces={current?.workspaces ?? []}
        onMode={chooseShellMode}
      />
      <main
        className={`agent-main ${
          composerCentered ? "composer-centered" : ""
        }`}
      >
        <AgentHeader
          current={current}
          inspector={inspector}
          navigationOpen={navigationOpen}
          onInspector={setInspector}
          onMenu={() => setNavigationOpen(true)}
          onMode={chooseShellMode}
          onReconcile={reconcile}
          operation={operation}
          workspace={workspace}
        />
        {error && (
          <div className="shell-error" role="alert">
            <AlertTriangle size={16} aria-hidden="true" />
            <span>{error}</span>
            <button
              type="button"
              aria-label="关闭错误提示"
              title="关闭错误提示"
              onClick={() => setError(null)}
            >
              <X size={15} aria-hidden="true" />
            </button>
          </div>
        )}
        <section
          className="agent-workspace"
          id="agent-workspace"
          aria-label={WORKSPACE_LABELS[workspace]}
        >
          <WorkspaceContent
            current={current}
            member={member}
            onMember={openMember}
            onQuote={quoteAgentOutput}
            onReviewChange={onReviewChange}
            onWorkspace={chooseWorkspace}
            source={source}
            workspace={workspace}
          />
        </section>
        {workspace === "CONVERSATION" && (
          <Composer
            busy={busy}
            contexts={composerQuotes}
            current={current}
            draft={draft}
            mode={composerMode}
            modelSelection={modelSelection}
            models={composerModels}
            onAddCustomModel={addCustomModel}
            onAdmit={admitSource}
            onChange={setDraft}
            onDirectory={selectDirectory}
            onFiles={selectFiles}
            onGoalTemplate={insertGoalTemplate}
            onError={setError}
            onModelSelection={chooseModel}
            onRemoveFile={(name) =>
              setPendingFiles((files) =>
                files.filter((file) => file.name !== name),
              )
            }
            onRemoveQuote={(messageId) =>
              setComposerQuotes((quotes) =>
                quotes.filter(
                  (quote) => quote.messageId !== messageId,
                ),
              )
            }
            onMode={setComposerMode}
            onDeleteCustomModel={deleteCustomModel}
            onSend={sendMessage}
            onKeyDown={onComposerKeyDown}
            pendingFiles={pendingFiles}
            source={source}
            sourceContract={sourceContract}
          />
        )}
        <ShellStatusBar
          connection={connection}
          current={current}
          notice={operation ?? notice}
        />
      </main>
      {inspector && current && (
        <AgentInspector
          current={current}
          modal={inspectorModal}
          member={member}
          mode={inspector}
          onClose={() => setInspector(null)}
          onMember={openMember}
        />
      )}
      {sessionToClose && (
        <CloseSessionDialog
          busy={busy}
          session={sessionToClose}
          onCancel={dismissSessionClose}
          onConfirm={closeSession}
        />
      )}
    </div>
  );
}

function SessionRail({
  currentId,
  modal,
  mobileOpen,
  onClose,
  onCreate,
  onDelete,
  onMode,
  onOpen,
  onQuery,
  query,
  sessions,
  workspace,
  workspaces,
}: {
  currentId: string | null;
  modal: boolean;
  mobileOpen: boolean;
  onClose: () => void;
  onCreate: () => void;
  onDelete: (session: AgentShellSessionSummary) => void;
  onMode: (mode: ShellMode) => void;
  onOpen: (sessionId: string) => void;
  onQuery: (value: string) => void;
  query: string;
  sessions: AgentShellSessionSummary[];
  workspace: AgentShellWorkspaceKind;
  workspaces: AgentShellWorkspaceDescriptor[];
}) {
  return (
    <aside
      id="session-navigation"
      className={`session-rail ${mobileOpen ? "mobile-open" : ""}`}
      aria-label="会话导航"
      aria-modal={modal || undefined}
      role={modal ? "dialog" : undefined}
    >
      <header className="shell-brand">
        <span className="shell-brand-mark" aria-hidden="true">
          E
        </span>
        <span>
          <strong>Eval Dataset</strong>
          <small>Agent Workbench</small>
        </span>
        <button
          type="button"
          className="shell-icon-button shell-mobile-only"
          onClick={onClose}
          aria-label="关闭会话导航"
          title="关闭会话导航"
        >
          <X size={17} aria-hidden="true" />
        </button>
      </header>
      <button
        type="button"
        className="new-session-button"
        onClick={onCreate}
      >
        <Plus size={16} aria-hidden="true" />
        新建会话
      </button>
      <label className="session-search">
        <Search size={14} aria-hidden="true" />
        <span className="sr-only">搜索会话</span>
        <input
          autoFocus
          type="search"
          name="session-search"
          autoComplete="off"
          value={query}
          onChange={(event) => onQuery(event.target.value)}
          placeholder="搜索会话…"
        />
      </label>
      <div className="session-list">
        <div className="rail-section-label">
          <span>会话</span>
          <small>{sessions.length}</small>
        </div>
        {sessions.map((session) => (
          <div
            className={`session-row-shell ${
              session.session_id === currentId ? "active" : ""
            }`}
            key={session.session_id}
          >
            <button
              type="button"
              className="session-row"
              onClick={() => onOpen(session.session_id)}
              aria-current={
                session.session_id === currentId ? "page" : undefined
              }
            >
              <MessageSquare size={15} aria-hidden="true" />
              <span>
                <strong>{session.title ?? "新会话"}</strong>
                <small>{formatRelativeTime(session.updated_at)}</small>
              </span>
            </button>
            <button
              type="button"
              className="session-delete-button"
              data-session-id={session.session_id}
              onClick={() => onDelete(session)}
              aria-label={`删除会话 ${session.title ?? "新会话"}`}
              title="删除会话"
            >
              <Trash2 size={14} aria-hidden="true" />
            </button>
          </div>
        ))}
        {!sessions.length && (
          <p className="rail-empty">暂无会话</p>
        )}
      </div>
      <nav className="workspace-navigation" aria-label="工作区">
        <div className="rail-section-label">
          <span>工作区</span>
        </div>
        {SHELL_MODES.map((item) => {
          const active = shellModeForWorkspace(workspace) === item.mode;
          const itemCount = modeItemCount(item.mode, workspaces);
          const Icon = item.icon;
          return (
            <button
              type="button"
              key={item.mode}
              className={active ? "active" : ""}
              onClick={() => onMode(item.mode)}
              aria-current={active ? "page" : undefined}
              title={item.label}
            >
              <Icon size={15} aria-hidden="true" />
              <span className="workspace-mode-copy">
                <strong>{item.label}</strong>
                <small>
                  {item.mode === "CONVERSATION"
                    ? "目标与上下文"
                    : item.mode === "WORKBENCH"
                      ? "来源、准则与交付"
                      : "Team、Graph 与活动"}
                </small>
              </span>
              <small className="workspace-state">
                {itemCount || ""}
              </small>
            </button>
          );
        })}
      </nav>
      <footer className="shell-principal">
        <ShieldCheck size={15} aria-hidden="true" />
        <span>
          <small>本地控制面</small>
          <code>Balanced Autonomy</code>
        </span>
      </footer>
    </aside>
  );
}

function CloseSessionDialog({
  busy,
  onCancel,
  onConfirm,
  session,
}: {
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  session: AgentShellSessionSummary;
}) {
  return (
    <div className="session-dialog-layer">
      <button
        type="button"
        className="session-dialog-backdrop"
        onClick={onCancel}
        disabled={busy}
        aria-label="关闭删除会话确认"
      />
      <section
        id="close-session-dialog"
        className="close-session-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="close-session-title"
        aria-describedby="close-session-description"
      >
        <span className="close-session-glyph" aria-hidden="true">
          <Trash2 size={19} />
        </span>
        <div>
          <h2 id="close-session-title">删除会话？</h2>
          <p id="close-session-description">
            “{session.title ?? "新会话"}”将从会话列表移除，评测审计记录仍会保留。
          </p>
        </div>
        <footer>
          <button
            type="button"
            className="dialog-cancel-button"
            onClick={onCancel}
            disabled={busy}
          >
            取消
          </button>
          <button
            type="button"
            className="dialog-delete-button"
            onClick={onConfirm}
            disabled={busy}
          >
            <Trash2 size={14} aria-hidden="true" />
            {busy ? "正在删除" : "删除会话"}
          </button>
        </footer>
      </section>
    </div>
  );
}

function AgentHeader({
  current,
  inspector,
  navigationOpen,
  onInspector,
  onMenu,
  onMode,
  onReconcile,
  operation,
  workspace,
}: {
  current: AgentShellProjection | null;
  inspector: InspectorView;
  navigationOpen: boolean;
  onInspector: (value: InspectorView) => void;
  onMenu: () => void;
  onMode: (mode: ShellMode) => void;
  onReconcile: () => void;
  operation: string | null;
  workspace: AgentShellWorkspaceKind;
}) {
  const needsSync =
    current?.execution_phase === "RUNNING" ||
    current?.execution_phase === "WAITING_REVIEW" ||
    current?.execution_phase === "VERIFICATION_REQUIRED";
  return (
    <header className="agent-header">
      <button
        type="button"
        className="shell-icon-button shell-mobile-only"
        onClick={onMenu}
        aria-label="打开会话导航"
        title="打开会话导航"
        aria-controls="session-navigation"
        aria-expanded={navigationOpen}
      >
        <Menu size={18} aria-hidden="true" />
      </button>
      <div className="agent-product-lockup">
        <span className="agent-product-seal" aria-hidden="true">
          E
        </span>
        <strong>Eval Dataset Agent</strong>
      </div>
      <nav className="agent-mode-tabs" aria-label="主要视图">
        {SHELL_MODES.map(({ mode, compactLabel, icon: Icon }) => (
          <button
            type="button"
            key={mode}
            className={
              shellModeForWorkspace(workspace) === mode ? "active" : ""
            }
            onClick={() => onMode(mode)}
            aria-current={
              shellModeForWorkspace(workspace) === mode
                ? "page"
                : undefined
            }
          >
            <Icon size={14} aria-hidden="true" />
            <span>{compactLabel}</span>
          </button>
        ))}
      </nav>
      <div className="agent-heading">
        <span>{current ? WORKSPACE_LABELS[workspace] : "对话"}</span>
        <strong>{current?.session.title ?? "新的评测任务"}</strong>
      </div>
      <div className="header-actions">
        {needsSync && (
          <button
            type="button"
            className="shell-icon-button"
            onClick={onReconcile}
            disabled={operation !== null}
            aria-label="同步执行状态"
            title="同步执行状态"
          >
            <RefreshCw
              size={16}
              className={operation ? "spin" : ""}
              aria-hidden="true"
            />
          </button>
        )}
        <button
          type="button"
          className={`shell-icon-button ${inspector === "team" ? "active" : ""}`}
          onClick={() =>
            onInspector(inspector === "team" ? null : "team")
          }
          disabled={!current?.team}
          aria-label="团队"
          title="团队"
          aria-controls="agent-inspector"
          aria-expanded={inspector === "team"}
        >
          <Users size={16} aria-hidden="true" />
        </button>
        <button
          type="button"
          className={`shell-icon-button ${
            inspector === "activity" ? "active" : ""
          }`}
          onClick={() =>
            onInspector(
              inspector === "activity" ? null : "activity",
            )
          }
          disabled={!current}
          aria-label="活动"
          title="活动"
          aria-controls="agent-inspector"
          aria-expanded={inspector === "activity"}
        >
          <Activity size={16} aria-hidden="true" />
          {!!current?.activity.length && (
            <span className="icon-count">{current.activity.length}</span>
          )}
        </button>
      </div>
    </header>
  );
}

function WorkspaceContent({
  current,
  member,
  onMember,
  onQuote,
  onReviewChange,
  onWorkspace,
  source,
  workspace,
}: {
  current: AgentShellProjection | null;
  member: AgentShellMemberProjection | null;
  onMember: (member: AgentShellTeamMemberSummary) => void;
  onQuote: (messageId: string, content: string) => void;
  onReviewChange: (view: PlanReviewView) => void;
  onWorkspace: (kind: AgentShellWorkspaceKind) => void;
  source: AgentShellSourceAdmission | null;
  workspace: AgentShellWorkspaceKind;
}) {
  if (!current) return <EmptyConversation />;
  if (workspace === "CONVERSATION") {
    return (
      <Conversation
        current={current}
        onQuote={onQuote}
        onWorkspace={onWorkspace}
        source={source}
      />
    );
  }
  let content: ReactNode;
  if (workspace === "PLAN_REVIEW") {
    content = (
      <div className="embedded-review-shell">
        <PlanReviewWorkspace
          embedded
          initialReviewId={
            current.plan_reviews[0]?.request_ref.object_id ?? null
          }
          onReviewChange={onReviewChange}
          principal={PRINCIPAL}
        />
      </div>
    );
  } else if (workspace === "TEAM") {
    content = (
      <TeamWorkspace
        current={current}
        member={member}
        onMember={onMember}
      />
    );
  } else if (workspace === "ACTIVITY") {
    content = <ActivityTimeline events={current.activity} />;
  } else if (workspace === "TRACE") {
    content = <TraceWorkspace current={current} source={source} />;
  } else if (workspace === "DELIVERY") {
    content = <DeliveryWorkspace current={current} />;
  } else {
    content = (
      <SummaryWorkspace
        current={current}
        kind={workspace}
      />
    );
  }

  return (
    <div className="grouped-workspace">
      <WorkspaceSubnav
        current={current}
        onWorkspace={onWorkspace}
        workspace={workspace}
      />
      <div
        className={`grouped-workspace-body ${
          workspace === "PLAN_REVIEW" ? "review-body" : ""
        }`}
      >
        {content}
      </div>
    </div>
  );
}

function WorkspaceSubnav({
  current,
  onWorkspace,
  workspace,
}: {
  current: AgentShellProjection;
  onWorkspace: (kind: AgentShellWorkspaceKind) => void;
  workspace: AgentShellWorkspaceKind;
}) {
  const mode = shellModeForWorkspace(workspace);
  const kinds =
    mode === "RUN" ? RUN_WORKSPACES : WORKBENCH_WORKSPACES;
  const label = mode === "RUN" ? "运行记录视图" : "评测工作台视图";
  return (
    <nav className="workspace-subnav" aria-label={label}>
      {kinds.map((kind) => {
        const descriptor = current.workspaces.find(
          (item) => item.kind === kind,
        );
        const Icon = WORKSPACE_ICONS[kind];
        const blocked = descriptor?.status === "BLOCKED";
        return (
          <button
            type="button"
            key={kind}
            className={workspace === kind ? "active" : ""}
            onClick={() => onWorkspace(kind)}
            disabled={blocked}
            data-workspace-status={descriptor?.status ?? "UNAVAILABLE"}
            aria-current={workspace === kind ? "page" : undefined}
            title={
              blocked
                ? descriptor.reason_codes.join(", ")
                : WORKSPACE_LABELS[kind]
            }
          >
            <Icon size={14} aria-hidden="true" />
            <span>{WORKSPACE_LABELS[kind]}</span>
            {!!descriptor?.item_count && (
              <small>{descriptor.item_count}</small>
            )}
          </button>
        );
      })}
    </nav>
  );
}

function EmptyConversation() {
  return (
    <div className="conversation-empty">
      <div className="empty-glyph" aria-hidden="true">
        <Boxes size={24} />
      </div>
      <h1>新建评测任务</h1>
      <p>创建会话后即可提交需求。</p>
    </div>
  );
}

function Conversation({
  current,
  onQuote,
  onWorkspace,
  source,
}: {
  current: AgentShellProjection;
  onQuote: (messageId: string, content: string) => void;
  onWorkspace: (kind: AgentShellWorkspaceKind) => void;
  source: AgentShellSourceAdmission | null;
}) {
  return (
    <div className="conversation-view">
      <section className="conversation-intro">
        <div>
          <span className="eyebrow">Evaluation Dataset Agent</span>
          <h1>{current.session.title ?? "未命名会话"}</h1>
        </div>
        <PhaseBadge phase={current.execution_phase} />
      </section>
      <div className="conversation-meta" aria-label="会话状态">
        <span>
          <Database size={13} aria-hidden="true" />
          authority v{current.session.session_version}
        </span>
        <span>
          <GitBranch size={13} aria-hidden="true" />
          {current.graph
            ? `transition ${current.graph.transition_number}`
            : "尚未创建 Graph"}
        </span>
        <span>
          <Paperclip size={13} aria-hidden="true" />
          {source ? `${source.source_count} 条来源` : "未接入来源"}
        </span>
        <span>
          <ShieldCheck size={13} aria-hidden="true" />
          {current.evidence_class ?? "等待证据"}
        </span>
      </div>
      {current.team && (
        <button
          type="button"
          className="conversation-team-strip"
          onClick={() => onWorkspace("TEAM")}
        >
          <span className="team-strip-icon" aria-hidden="true">
            <Users size={15} />
          </span>
          <span className="team-strip-copy">
            <strong>Agent Team</strong>
            <small>
              {current.team.members.length} 个成员 ·{" "}
              {current.team.tasks.filter(
                (task) => task.status === "COMPLETED",
              ).length}
              /{current.team.tasks.length} 个任务完成
            </small>
          </span>
          <span className="team-avatar-stack" aria-hidden="true">
            {current.team.members.slice(0, 4).map((member) => (
              <span key={member.member_id}>
                {member.is_coordinator
                  ? "C"
                  : member.role.slice(0, 1).toUpperCase()}
              </span>
            ))}
          </span>
          <ChevronRight size={15} aria-hidden="true" />
        </button>
      )}
      <div className="transcript" aria-live="polite">
        {!current.transcript.length && (
          <div className="conversation-zero">
            <Bot size={20} aria-hidden="true" />
            <strong>等待评测需求</strong>
          </div>
        )}
        {current.transcript.map((entry) => (
          <article
            className={`message-row role-${entry.role.toLowerCase()}`}
            key={entry.message_id}
          >
            <span className="message-avatar" aria-hidden="true">
              {entry.role === "USER" ? (
                <User size={15} />
              ) : (
                <Bot size={15} />
              )}
            </span>
            <div className="message-content">
              <header>
                <strong>{entry.role === "USER" ? "你" : "Eval Agent"}</strong>
                <time dateTime={entry.created_at}>
                  {formatClock(entry.created_at)}
                </time>
              </header>
              <p>{entry.content}</p>
              {!!entry.artifact_envelope_refs.length && (
                <span className="message-attachments">
                  <Paperclip size={12} aria-hidden="true" />
                  {entry.artifact_envelope_refs.length} 个来源引用
                </span>
              )}
              {entry.role === "ASSISTANT" && (
                <button
                  type="button"
                  className="message-quote-action"
                  onClick={(event) =>
                    onQuote(
                      entry.message_id,
                      selectedMessageContent(event, entry.content),
                    )
                  }
                  aria-label="引用 Agent 输出到输入框"
                  title="引用 Agent 输出到输入框"
                >
                  <Quote size={13} aria-hidden="true" />
                  引用
                </button>
              )}
            </div>
          </article>
        ))}
        {current.pending_interactions.map((interaction) => (
          <InteractionCard
            interaction={interaction}
            key={interaction.interaction_id}
            onOpen={() =>
              onWorkspace(
                interaction.kind === "PLAN_REVIEW"
                  ? "PLAN_REVIEW"
                  : "ACTIVITY",
              )
            }
          />
        ))}
      </div>
    </div>
  );
}

function InteractionCard({
  interaction,
  onOpen,
}: {
  interaction: AgentShellInteractionCard;
  onOpen: () => void;
}) {
  const Icon =
    interaction.kind === "PLAN_REVIEW"
      ? ListChecks
      : interaction.kind === "VERIFICATION"
        ? ShieldCheck
        : AlertTriangle;
  return (
    <article className={`interaction-card kind-${interaction.kind.toLowerCase()}`}>
      <Icon size={18} aria-hidden="true" />
      <div>
        <span>{interactionLabel(interaction.kind)}</span>
        <strong>{interactionTitle(interaction.kind)}</strong>
        {interaction.questions.map((question) => (
          <p key={question}>{question}</p>
        ))}
        {!!interaction.reason_codes.length && (
          <div className="reason-list">
            {interaction.reason_codes.map((reason) => (
              <code key={reason}>{reason}</code>
            ))}
          </div>
        )}
        <button
          type="button"
          className="interaction-action"
          onClick={onOpen}
        >
          {interaction.kind === "PLAN_REVIEW"
            ? "打开审核"
            : "查看详情"}
          <ChevronRight size={14} aria-hidden="true" />
        </button>
      </div>
    </article>
  );
}

function ModelPicker({
  modelSelection,
  models,
  onAddCustomModel,
  onClose,
  onDeleteCustomModel,
  onSelect,
}: {
  modelSelection: ModelSelection;
  models: readonly ComposerModelOption[];
  onAddCustomModel: (draft: CustomModelDraft) => string | null;
  onClose: () => void;
  onDeleteCustomModel: (modelId: string) => void;
  onSelect: (selection: ModelSelection) => void;
}) {
  const compact = useMediaQuery("(max-width: 600px)");
  const [query, setQuery] = useState("");
  const [view, setView] = useState<"LIST" | "CUSTOM">("LIST");
  const [formError, setFormError] = useState<string | null>(null);
  const [customDraft, setCustomDraft] = useState<CustomModelDraft>({
    displayName: "",
    providerName: "",
    modelId: "",
    protocol: "OPENAI_RESPONSES",
    baseUrl: "",
    credentialEnvVar: "",
    maximumEffort: "HIGH",
  });
  const selectedModel =
    models.find((model) => model.id === modelSelection.modelId) ??
    models[0] ??
    BUILT_IN_MODELS[0];
  const visibleModels = models.filter((model) => {
    const needle = query.trim().toLocaleLowerCase();
    if (!needle) return true;
    return `${model.displayName} ${model.providerName} ${model.modelId}`
      .toLocaleLowerCase()
      .includes(needle);
  });
  const effortIndex = Math.max(
    0,
    selectedModel.supportedEfforts.indexOf(modelSelection.effort),
  );
  const effortProgress =
    selectedModel.supportedEfforts.length <= 1
      ? 0
      : (effortIndex / (selectedModel.supportedEfforts.length - 1)) *
        100;

  useEffect(() => {
    const onEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    document.addEventListener("keydown", onEscape);
    return () => document.removeEventListener("keydown", onEscape);
  }, [onClose]);

  function submitCustomModel(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    const error = onAddCustomModel(customDraft);
    if (error) {
      setFormError(error);
      return;
    }
    setFormError(null);
    setView("LIST");
    setQuery("");
  }

  function renderPicker(content: ReactNode) {
    return compact ? createPortal(content, document.body) : content;
  }

  if (view === "CUSTOM") {
    return renderPicker(
      <form
        className="model-picker-popover custom-model-form"
        role="dialog"
        aria-label="添加自定义模型"
        onSubmit={submitCustomModel}
      >
        <header>
          <button
            type="button"
            onClick={() => {
              setFormError(null);
              setView("LIST");
            }}
            aria-label="返回模型列表"
            title="返回"
          >
            <ArrowLeft size={15} aria-hidden="true" />
          </button>
          <span>
            <strong>添加自定义模型</strong>
            <small>配置模型端点，不保存明文密钥</small>
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭模型选择"
            title="关闭"
          >
            <X size={14} aria-hidden="true" />
          </button>
        </header>
        <div className="custom-model-fields">
          <label>
            <span>显示名称</span>
            <input
              autoFocus
              required
              maxLength={60}
              value={customDraft.displayName}
              onChange={(event) =>
                setCustomDraft((value) => ({
                  ...value,
                  displayName: event.target.value,
                }))
              }
              placeholder="例如：团队代码模型"
            />
          </label>
          <label>
            <span>服务商</span>
            <input
              required
              maxLength={60}
              value={customDraft.providerName}
              onChange={(event) =>
                setCustomDraft((value) => ({
                  ...value,
                  providerName: event.target.value,
                }))
              }
              placeholder="例如：OpenRouter"
            />
          </label>
          <label>
            <span>API 协议</span>
            <select
              value={customDraft.protocol}
              onChange={(event) =>
                setCustomDraft((value) => ({
                  ...value,
                  protocol: event.target
                    .value as CustomModelDraft["protocol"],
                }))
              }
            >
              <option value="OPENAI_RESPONSES">OpenAI Responses</option>
              <option value="OPENAI_CHAT_COMPLETIONS">
                OpenAI Chat Completions
              </option>
              <option value="ANTHROPIC_MESSAGES">
                Anthropic Messages
              </option>
            </select>
          </label>
          <label>
            <span>模型 ID</span>
            <input
              required
              maxLength={128}
              spellCheck={false}
              value={customDraft.modelId}
              onChange={(event) =>
                setCustomDraft((value) => ({
                  ...value,
                  modelId: event.target.value,
                }))
              }
              placeholder="provider/model-name"
            />
          </label>
          <label className="custom-model-field-wide">
            <span>Base URL</span>
            <input
              required
              inputMode="url"
              spellCheck={false}
              value={customDraft.baseUrl}
              onChange={(event) =>
                setCustomDraft((value) => ({
                  ...value,
                  baseUrl: event.target.value,
                }))
              }
              placeholder="https://gateway.example.com/v1"
            />
          </label>
          <label>
            <span>凭证环境变量</span>
            <input
              maxLength={64}
              spellCheck={false}
              value={customDraft.credentialEnvVar}
              onChange={(event) =>
                setCustomDraft((value) => ({
                  ...value,
                  credentialEnvVar: event.target.value.toUpperCase(),
                }))
              }
              placeholder="CUSTOM_API_KEY"
            />
          </label>
          <label>
            <span>最高思考深度</span>
            <select
              value={customDraft.maximumEffort}
              onChange={(event) =>
                setCustomDraft((value) => ({
                  ...value,
                  maximumEffort: event.target
                    .value as ReasoningEffort,
                }))
              }
            >
              <option value="NONE">不支持</option>
              <option value="LOW">低</option>
              <option value="MEDIUM">中</option>
              <option value="HIGH">高</option>
              <option value="XHIGH">极高</option>
              <option value="MAX">Max</option>
            </select>
          </label>
        </div>
        {formError && (
          <p className="custom-model-error" role="alert">
            {formError}
          </p>
        )}
        <p className="custom-model-security-note">
          <ShieldCheck size={14} aria-hidden="true" />
          仅保存环境变量名。请在 Agent Host 环境中配置真实凭证。
        </p>
        <footer>
          <button
            type="button"
            className="dialog-cancel-button"
            onClick={() => setView("LIST")}
          >
            取消
          </button>
          <button type="submit" className="model-save-button">
            添加模型
          </button>
        </footer>
      </form>,
    );
  }

  return renderPicker(
    <section
      className="model-picker-popover"
      role="dialog"
      aria-label="模型和思考深度"
    >
      <header>
        <span>
          <strong>模型</strong>
          <small>选择模型后调整对应思考深度</small>
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="关闭模型选择"
          title="关闭"
        >
          <X size={14} aria-hidden="true" />
        </button>
      </header>
      <label className="model-picker-search">
        <Search size={14} aria-hidden="true" />
        <span className="sr-only">搜索模型</span>
        <input
          autoFocus
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索模型或服务商"
        />
      </label>
      <div className="model-picker-list" role="listbox">
        {visibleModels.map((model) => {
          const selected = model.id === selectedModel.id;
          return (
            <div
              className={`model-option-shell ${
                selected ? "selected" : ""
              }`}
              key={model.id}
            >
              <button
                type="button"
                className="model-option"
                role="option"
                aria-selected={selected}
                onClick={() =>
                  onSelect(
                    selectionForModel(
                      model,
                      selected ? modelSelection.effort : undefined,
                    ),
                  )
                }
              >
                <span className="model-provider-mark" aria-hidden="true">
                  {model.displayName.slice(0, 1).toLocaleUpperCase()}
                </span>
                <span>
                  <strong>{model.displayName}</strong>
                  <small>
                    {model.providerName} · {model.modelId}
                  </small>
                </span>
                {model.source === "CUSTOM" && <em>自定义</em>}
                {selected && <Check size={15} aria-hidden="true" />}
              </button>
              {model.source === "CUSTOM" && (
                <button
                  type="button"
                  className="model-option-delete"
                  aria-label={`删除自定义模型 ${model.displayName}`}
                  title="删除自定义模型"
                  onClick={() => onDeleteCustomModel(model.id)}
                >
                  <Trash2 size={13} aria-hidden="true" />
                </button>
              )}
            </div>
          );
        })}
        {!visibleModels.length && (
          <p className="model-picker-empty">没有匹配的模型</p>
        )}
      </div>
      <section className="model-effort-control">
        <header>
          <span>思考深度</span>
          <strong>{EFFORT_LABELS[modelSelection.effort]}</strong>
        </header>
        <input
          type="range"
          min={0}
          max={Math.max(0, selectedModel.supportedEfforts.length - 1)}
          step={1}
          value={effortIndex}
          aria-label={`${selectedModel.displayName} 思考深度`}
          aria-valuetext={EFFORT_LABELS[modelSelection.effort]}
          onChange={(event) => {
            const effort =
              selectedModel.supportedEfforts[
                Number(event.target.value)
              ] ?? selectedModel.defaultEffort;
            onSelect({
              modelId: selectedModel.id,
              effort,
            });
          }}
          style={
            {
              "--model-effort-progress": `${effortProgress}%`,
            } as CSSProperties
          }
          disabled={selectedModel.supportedEfforts.length <= 1}
        />
        <div className="model-effort-labels" aria-hidden="true">
          {selectedModel.supportedEfforts.map((effort) => (
            <span
              className={
                effort === modelSelection.effort ? "active" : ""
              }
              key={effort}
            >
              {EFFORT_LABELS[effort]}
            </span>
          ))}
        </div>
      </section>
      <footer className="model-picker-footer">
        <span>
          <ShieldCheck size={13} aria-hidden="true" />
          按会话保存，执行仍由 Host 策略校验
        </span>
        <button
          type="button"
          onClick={() => {
            setFormError(null);
            setView("CUSTOM");
          }}
        >
          <Settings2 size={14} aria-hidden="true" />
          自定义模型
        </button>
      </footer>
    </section>,
  );
}

function Composer({
  busy,
  contexts,
  current,
  draft,
  mode,
  modelSelection,
  models,
  onAddCustomModel,
  onAdmit,
  onChange,
  onDeleteCustomModel,
  onDirectory,
  onError,
  onFiles,
  onGoalTemplate,
  onKeyDown,
  onMode,
  onModelSelection,
  onRemoveFile,
  onRemoveQuote,
  onSend,
  pendingFiles,
  source,
  sourceContract,
}: {
  busy: boolean;
  contexts: ComposerQuote[];
  current: AgentShellProjection | null;
  draft: string;
  mode: ComposerMode;
  modelSelection: ModelSelection;
  models: readonly ComposerModelOption[];
  onAddCustomModel: (draft: CustomModelDraft) => string | null;
  onAdmit: () => void;
  onChange: (value: string) => void;
  onDeleteCustomModel: (modelId: string) => void;
  onDirectory: (event: ChangeEvent<HTMLInputElement>) => void;
  onError: (message: string | null) => void;
  onFiles: (event: ChangeEvent<HTMLInputElement>) => void;
  onGoalTemplate: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
  onMode: (mode: ComposerMode) => void;
  onModelSelection: (selection: ModelSelection) => void;
  onRemoveFile: (name: string) => void;
  onRemoveQuote: (messageId: string) => void;
  onSend: () => void;
  pendingFiles: File[];
  source: AgentShellSourceAdmission | null;
  sourceContract: AgentShellSourceContract | null;
}) {
  const [panel, setPanel] = useState<ComposerPanel>(null);
  const [commandIndex, setCommandIndex] = useState(0);
  const [sourceChoicesOpen, setSourceChoicesOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const modelTriggerRef = useRef<HTMLButtonElement | null>(null);
  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const manifestSelected = pendingFiles.some(
    (file) => file.name === "manifest.csv",
  );
  const tracesSelected = pendingFiles.filter((file) =>
    file.name.endsWith(".jsonl"),
  ).length;
  const canAdmit =
    !!current &&
    !source &&
    manifestSelected &&
    tracesSelected > 0 &&
    !busy;
  const commandMatch = draft.match(/^\s*\/([^\s]*)$/);
  const commandQuery = commandMatch?.[1].toLocaleLowerCase() ?? null;
  const matchingCommands =
    commandQuery === null
      ? []
      : COMPOSER_COMMANDS.filter(
          (item) =>
            item.command.slice(1).includes(commandQuery) ||
            item.label.toLocaleLowerCase().includes(commandQuery),
        );
  const composedLength = composeMessageContent(draft, contexts).length;
  const selectedModel =
    models.find((model) => model.id === modelSelection.modelId) ??
    models[0] ??
    BUILT_IN_MODELS[0];

  useEffect(() => {
    setCommandIndex(0);
  }, [commandQuery]);

  useEffect(
    () => () => {
      if (!recognitionRef.current) return;
      recognitionRef.current.onend = null;
      recognitionRef.current.stop();
    },
    [],
  );

  function focusInput() {
    window.requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function closeModelPicker() {
    setPanel(null);
    window.requestAnimationFrame(() => modelTriggerRef.current?.focus());
  }

  function applyCommand(command: string) {
    if (command === "/goal") {
      onGoalTemplate();
      setPanel(null);
    } else if (command === "/plan") {
      onMode("PLAN");
      onChange("");
      setPanel(null);
    } else if (command === "/agent") {
      onMode("AGENT");
      onChange("");
      setPanel(null);
    } else if (command === "/model") {
      setPanel("MODEL");
      onChange("");
    } else if (command === "/source") {
      setPanel("ADD");
      setSourceChoicesOpen(true);
      onChange("");
    } else if (command === "/skills") {
      setPanel("EXTENSIONS");
      onChange("");
    }
    focusInput();
  }

  function toggleVoiceInput() {
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    const Recognition = speechRecognitionConstructor();
    if (!Recognition) {
      onError("当前浏览器不支持语音输入。");
      return;
    }
    const recognition = new Recognition();
    recognition.lang = "zh-CN";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onresult = (event) => {
      let transcript = "";
      for (
        let index = event.resultIndex;
        index < event.results.length;
        index += 1
      ) {
        if (event.results[index]?.isFinal) {
          transcript += event.results[index][0]?.transcript ?? "";
        }
      }
      const value = transcript.trim();
      if (!value) return;
      onChange(
        draft.trim()
          ? `${draft.trimEnd()} ${value}`
          : value,
      );
      focusInput();
    };
    recognition.onerror = (event) => {
      if (event.error !== "aborted") {
        onError("语音输入失败，请检查麦克风权限后重试。");
      }
      setListening(false);
    };
    recognition.onend = () => {
      recognitionRef.current = null;
      setListening(false);
    };
    try {
      recognitionRef.current = recognition;
      recognition.start();
      setListening(true);
      onError(null);
    } catch {
      recognitionRef.current = null;
      setListening(false);
      onError("无法启动语音输入，请检查麦克风权限。");
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (commandQuery !== null && matchingCommands.length) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const direction = event.key === "ArrowDown" ? 1 : -1;
        setCommandIndex(
          (value) =>
            (value + direction + matchingCommands.length) %
            matchingCommands.length,
        );
        return;
      }
      if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !event.nativeEvent.isComposing
      ) {
        event.preventDefault();
        const selected =
          matchingCommands[commandIndex] ?? matchingCommands[0];
        applyCommand(selected.command);
        return;
      }
    }
    if (event.key === "Escape" && panel) {
      event.preventDefault();
      setPanel(null);
      return;
    }
    onKeyDown(event);
  }

  return (
    <section className="composer-wrap" aria-label="消息编辑器">
      {panel === "MODEL" && (
        <ModelPicker
          modelSelection={modelSelection}
          models={models}
          onAddCustomModel={onAddCustomModel}
          onClose={closeModelPicker}
          onDeleteCustomModel={onDeleteCustomModel}
          onSelect={onModelSelection}
        />
      )}
      {panel === "ADD" && (
        <div
          className="composer-popover composer-add-menu"
          role="dialog"
          aria-label="添加上下文"
        >
          <header>
            <strong>添加</strong>
            <button
              type="button"
              onClick={() => {
                setPanel(null);
                setSourceChoicesOpen(false);
              }}
              aria-label="关闭添加上下文"
              title="关闭"
            >
              <X size={14} aria-hidden="true" />
            </button>
          </header>
          <button
            type="button"
            className="composer-menu-item"
            onClick={() => setSourceChoicesOpen((value) => !value)}
            aria-expanded={sourceChoicesOpen}
          >
            <FolderOpen size={16} aria-hidden="true" />
            <span>
              <strong>文件和文件夹</strong>
              <small>
                {sourceContract
                  ? `最多 ${sourceContract.max_source_files} 个 JSONL`
                  : "添加 manifest.csv 与 JSONL 来源"}
              </small>
            </span>
            <ChevronRight
              className={sourceChoicesOpen ? "expanded" : ""}
              size={14}
              aria-hidden="true"
            />
          </button>
          {sourceChoicesOpen && (
            <div
              className="composer-source-choices"
              aria-label="文件和文件夹"
            >
              <label>
                <Paperclip size={14} aria-hidden="true" />
                <span>选择文件</span>
                <input
                  type="file"
                  aria-label="选择来源文件"
                  accept=".csv,.jsonl,text/csv,application/x-ndjson"
                  multiple
                  disabled={!current || !!source || busy}
                  onChange={(event) => {
                    onFiles(event);
                    setPanel(null);
                    setSourceChoicesOpen(false);
                  }}
                />
              </label>
              <label>
                <FolderOpen size={14} aria-hidden="true" />
                <span>选择文件夹</span>
                <input
                  type="file"
                  aria-label="选择来源目录"
                  multiple
                  {...({ webkitdirectory: "" } as Record<string, string>)}
                  disabled={!current || !!source || busy}
                  onChange={(event) => {
                    onDirectory(event);
                    setPanel(null);
                    setSourceChoicesOpen(false);
                  }}
                />
              </label>
            </div>
          )}
          <button
            type="button"
            className="composer-menu-item"
            disabled
            title="需要桌面 Host 与工作区权限契约"
          >
            <Boxes size={16} aria-hidden="true" />
            <span>
              <strong>在项目中使用</strong>
              <small>需要桌面 Host</small>
            </span>
          </button>
          <button
            type="button"
            className="composer-menu-item"
            onClick={() => {
              onGoalTemplate();
              setPanel(null);
            }}
          >
            <Target size={16} aria-hidden="true" />
            <span>
              <strong>目标</strong>
              <small>插入目标、验收标准和约束</small>
            </span>
          </button>
          <button
            type="button"
            className="composer-menu-item"
            onClick={() => applyCommand("/plan")}
          >
            <ListChecks size={16} aria-hidden="true" />
            <span>
              <strong>计划模式</strong>
              <small>提交后优先进入计划审核</small>
            </span>
          </button>
          <button
            type="button"
            className="composer-menu-item"
            onClick={() => applyCommand("/skills")}
          >
            <Plug size={16} aria-hidden="true" />
            <span>
              <strong>插件</strong>
              <small>查看当前会话可用能力</small>
            </span>
          </button>
        </div>
      )}
      {panel === "EXTENSIONS" && (
        <div
          className="composer-popover composer-extension-menu"
          role="dialog"
          aria-label="Skills 与插件"
        >
          <header>
            <span>
              <strong>插件</strong>
              <small>当前会话可用能力</small>
            </span>
            <button
              type="button"
              onClick={() => setPanel(null)}
              aria-label="关闭 Skills 与插件"
              title="关闭"
            >
              <X size={14} aria-hidden="true" />
            </button>
          </header>
          <div className="composer-extension-list">
            {COMPOSER_COMMANDS.slice(0, 3).map((item) => {
              const Icon = item.icon;
              return (
                <button
                  type="button"
                  key={item.command}
                  onClick={() => applyCommand(item.command)}
                >
                  <Icon size={15} aria-hidden="true" />
                  <span>
                    <strong>{item.label}</strong>
                    <code>{item.command}</code>
                  </span>
                </button>
              );
            })}
          </div>
          <div className="composer-extension-state" role="status">
            <Plug size={16} aria-hidden="true" />
            <span>
              <strong>插件注册表尚未接入</strong>
              <small>
                安装、权限审核和版本管理需要服务端契约
              </small>
            </span>
          </div>
        </div>
      )}
      {commandQuery !== null && !!matchingCommands.length && (
        <div
          className="composer-command-menu"
          role="listbox"
          aria-label="斜杠命令"
        >
          {matchingCommands.map((item, index) => {
            const Icon = item.icon;
            return (
              <button
                type="button"
                role="option"
                aria-selected={index === commandIndex}
                className={index === commandIndex ? "active" : ""}
                key={item.command}
                onClick={() => applyCommand(item.command)}
              >
                <Icon size={15} aria-hidden="true" />
                <span>
                  <strong>{item.label}</strong>
                  <small>{item.description}</small>
                </span>
                <code>{item.command}</code>
              </button>
            );
          })}
        </div>
      )}
      {!!contexts.length && (
        <div className="composer-contexts" aria-label="已引用的 Agent 输出">
          {contexts.map((context, index) => (
            <span key={`${context.messageId}-${index}`}>
              <Quote size={13} aria-hidden="true" />
              <span>
                <strong>Agent 输出</strong>
                <small>{compactText(context.content, 72)}</small>
              </span>
              <button
                type="button"
                onClick={() => onRemoveQuote(context.messageId)}
                aria-label="移除 Agent 输出引用"
                title="移除引用"
              >
                <X size={12} aria-hidden="true" />
              </button>
            </span>
          ))}
        </div>
      )}
      {!!pendingFiles.length && (
        <div className="pending-files">
          <div>
            {pendingFiles.map((file) => (
              <span key={file.name}>
                {file.name}
                <button
                  type="button"
                  aria-label={`移除 ${file.name}`}
                  title={`移除 ${file.name}`}
                  onClick={() => onRemoveFile(file.name)}
                >
                  <X size={12} aria-hidden="true" />
                </button>
              </span>
            ))}
          </div>
          <button
            type="button"
            className="admit-button"
            disabled={!canAdmit}
            onClick={onAdmit}
          >
            <UploadCloud size={15} aria-hidden="true" />
            确认接入
          </button>
        </div>
      )}
      <div className="composer">
        <label className="composer-input">
          <span className="sr-only">输入评测需求</span>
          <textarea
            ref={textareaRef}
            rows={1}
            name="evaluation-requirement"
            autoComplete="off"
            value={draft}
            disabled={!current || busy}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              current
                ? "描述评测目标、数据边界和交付要求…"
                : "请先创建一个会话…"
            }
          />
        </label>
        <footer className="composer-toolbar">
          <div className="composer-toolbar-left">
            <button
              type="button"
              className={`attach-button ${
                panel === "ADD" ? "active" : ""
              }`}
              onClick={() => {
                setSourceChoicesOpen(false);
                setPanel((value) =>
                  value === "ADD" ? null : "ADD",
                );
              }}
              disabled={!current || busy}
              aria-label="添加上下文"
              title="添加"
              aria-expanded={panel === "ADD"}
            >
              <Plus size={18} aria-hidden="true" />
            </button>
            <label className="composer-mode-control">
              {mode === "PLAN" ? (
                <ListChecks size={14} aria-hidden="true" />
              ) : (
                <Bot size={14} aria-hidden="true" />
              )}
              <span className="sr-only">交互模式</span>
              <select
                value={mode}
                onChange={(event) =>
                  onMode(event.target.value as ComposerMode)
                }
                disabled={!current || busy}
                title={
                  mode === "PLAN"
                    ? "提交后优先进入计划审核"
                    : "按 Balanced Autonomy 继续执行"
                }
              >
                <option value="AGENT">Agent 模式</option>
                <option value="PLAN">计划模式</option>
              </select>
              <ChevronDown size={12} aria-hidden="true" />
            </label>
          </div>
          <div className="composer-toolbar-right">
            <button
              ref={modelTriggerRef}
              type="button"
              className={`model-picker-trigger ${
                panel === "MODEL" ? "active" : ""
              }`}
              onClick={() =>
                panel === "MODEL"
                  ? closeModelPicker()
                  : setPanel("MODEL")
              }
              disabled={!current || busy}
              aria-label="选择模型和思考深度"
              aria-expanded={panel === "MODEL"}
              title="模型和思考深度"
            >
              <Bot size={14} aria-hidden="true" />
              <span>{selectedModel.displayName}</span>
              <small>{EFFORT_LABELS[modelSelection.effort]}</small>
              <ChevronDown size={12} aria-hidden="true" />
            </button>
            <button
              type="button"
              className={`voice-button ${listening ? "active" : ""}`}
              disabled={!current || busy}
              onClick={toggleVoiceInput}
              aria-label={listening ? "停止语音输入" : "语音输入"}
              aria-pressed={listening}
              title={listening ? "停止语音输入" : "语音输入"}
            >
              {listening ? (
                <Square size={14} aria-hidden="true" />
              ) : (
                <Mic size={17} aria-hidden="true" />
              )}
            </button>
            <button
              type="button"
              className="send-button"
              disabled={
                !current ||
                !composeMessageContent(draft, contexts) ||
                busy
              }
              onClick={onSend}
              aria-label="发送需求"
              title="发送需求"
            >
              <ArrowUp size={18} aria-hidden="true" />
            </button>
          </div>
        </footer>
        <span className="sr-only" aria-live="polite">
          已输入 {composedLength.toLocaleString()} / 32,768 字符
        </span>
      </div>
    </section>
  );
}

function TeamWorkspace({
  current,
  member,
  onMember,
}: {
  current: AgentShellProjection;
  member: AgentShellMemberProjection | null;
  onMember: (member: AgentShellTeamMemberSummary) => void;
}) {
  const team = current.team;
  if (!team) {
    return <UnavailableWorkspace title="团队尚未建立" />;
  }
  return (
    <div className="summary-workspace">
      <WorkspaceHeading
        eyebrow="Agent Team"
        title="协作团队"
        detail={`${team.members.length} 个成员 / ${team.tasks.length} 个任务`}
      />
      <div className="team-layout">
        <section className="team-members">
          <h2>成员</h2>
          {team.members.map((item) => (
            <button
              type="button"
              key={item.member_id}
              onClick={() => onMember(item)}
              className={
                member?.member.member_id === item.member_id
                  ? "active"
                  : ""
              }
            >
              <span className="member-avatar" aria-hidden="true">
                {item.is_coordinator ? "C" : item.role.slice(0, 1).toUpperCase()}
              </span>
              <span>
                <strong>{humanize(item.role)}</strong>
                <small>{shortIdentity(item.member_id, 34)}</small>
              </span>
              <StatusDot status={item.status} />
            </button>
          ))}
        </section>
        <section className="task-board">
          <h2>任务图</h2>
          {team.tasks.map((task) => (
            <article key={task.task_id}>
              <StatusDot status={task.status} />
              <div>
                <strong>{humanize(task.task_kind)}</strong>
                <small>{task.assigned_member_id ?? "等待分配"}</small>
              </div>
              <span>尝试 {task.attempt}</span>
            </article>
          ))}
        </section>
      </div>
    </div>
  );
}

function ActivityTimeline({
  events,
}: {
  events: AgentShellEvent[];
}) {
  return (
    <div className="summary-workspace">
      <WorkspaceHeading
        eyebrow="Committed Events"
        title="执行动态"
        detail={`${events.length} 条已提交事件`}
      />
      <ol className="event-timeline">
        {[...events].reverse().map((event) => (
          <li key={event.event_id}>
            <span className="event-sequence">{event.sequence}</span>
            <div>
              <strong>{humanize(event.event_kind)}</strong>
              <small>
                {humanize(event.family)}
                {event.runtime_id ? ` / ${event.runtime_id}` : ""}
              </small>
            </div>
            <time dateTime={event.occurred_at}>
              {formatClock(event.occurred_at)}
            </time>
          </li>
        ))}
      </ol>
    </div>
  );
}

function TraceWorkspace({
  current,
  source,
}: {
  current: AgentShellProjection;
  source: AgentShellSourceAdmission | null;
}) {
  if (!source) return <UnavailableWorkspace title="尚未接入轨迹来源" />;
  return (
    <div className="summary-workspace">
      <WorkspaceHeading
        eyebrow="Source Admission"
        title="轨迹来源"
        detail={`${source.source_count} 个 Trace / ${formatBytes(
          source.total_source_bytes,
        )}`}
      />
      <div className="source-ledger">
        {source.files.map((file) => (
          <article key={file.relative_name}>
            {file.kind === "MANIFEST" ? (
              <FileSpreadsheet size={17} aria-hidden="true" />
            ) : (
              <FileJson2 size={17} aria-hidden="true" />
            )}
            <div>
              <strong>{file.relative_name}</strong>
              <small>{file.media_type}</small>
            </div>
            <code>{file.sha256.slice(0, 12)}</code>
            <span>{formatBytes(file.size_bytes)}</span>
          </article>
        ))}
      </div>
      <p className="workspace-footnote">
        authority v{current.session.session_version} /{" "}
        {source.manifest_sha256.slice(0, 16)}
      </p>
    </div>
  );
}

function DeliveryWorkspace({
  current,
}: {
  current: AgentShellProjection;
}) {
  if (!current.delivery) {
    return <UnavailableWorkspace title="交付尚未生成" />;
  }
  return (
    <div className="summary-workspace">
      <WorkspaceHeading
        eyebrow="Candidate Delivery"
        title="候选数据集"
        detail={`${current.delivery.item_count} 个候选项`}
      />
      <dl className="delivery-metrics">
        <div>
          <dt>文件</dt>
          <dd>{current.delivery.file_count}</dd>
        </div>
        <div>
          <dt>总大小</dt>
          <dd>{formatBytes(current.delivery.total_bytes)}</dd>
        </div>
        <div>
          <dt>Bundle SHA</dt>
          <dd>{current.delivery.bundle_sha256.slice(0, 16)}</dd>
        </div>
      </dl>
    </div>
  );
}

function SummaryWorkspace({
  current,
  kind,
}: {
  current: AgentShellProjection;
  kind: AgentShellWorkspaceKind;
}) {
  const descriptor = current.workspaces.find(
    (item) => item.kind === kind,
  );
  if (!descriptor || descriptor.status === "EMPTY") {
    return (
      <UnavailableWorkspace
        title={`${WORKSPACE_LABELS[kind]}暂无内容`}
      />
    );
  }
  return (
    <div className="summary-workspace">
      <WorkspaceHeading
        eyebrow={humanize(descriptor.status)}
        title={WORKSPACE_LABELS[kind]}
        detail={`${descriptor.item_count} 个当前对象`}
      />
      <div className="owner-ref-list">
        {descriptor.owner_refs.map((reference) => (
          <article key={reference.object_id}>
            <Database size={15} aria-hidden="true" />
            <div>
              <strong>{humanize(reference.object_type)}</strong>
              <code title={reference.object_id}>
                {shortIdentity(reference.object_id, 46)}
              </code>
            </div>
            <span>{reference.object_version}</span>
          </article>
        ))}
      </div>
      {!!descriptor.reason_codes.length && (
        <div className="reason-list">
          {descriptor.reason_codes.map((reason) => (
            <code key={reason}>{reason}</code>
          ))}
        </div>
      )}
    </div>
  );
}

function WorkspaceHeading({
  detail,
  eyebrow,
  title,
}: {
  detail: string;
  eyebrow: string;
  title: string;
}) {
  return (
    <header className="summary-heading">
      <div>
        <span>{eyebrow}</span>
        <h1>{title}</h1>
      </div>
      <p>{detail}</p>
    </header>
  );
}

function UnavailableWorkspace({ title }: { title: string }) {
  return (
    <div className="workspace-empty" role="status">
      <Circle size={18} aria-hidden="true" />
      <strong>{title}</strong>
    </div>
  );
}

function AgentInspector({
  current,
  modal,
  member,
  mode,
  onClose,
  onMember,
}: {
  current: AgentShellProjection;
  modal: boolean;
  member: AgentShellMemberProjection | null;
  mode: Exclude<InspectorView, null>;
  onClose: () => void;
  onMember: (member: AgentShellTeamMemberSummary) => void;
}) {
  return (
    <aside
      id="agent-inspector"
      className="agent-inspector"
      role="dialog"
      aria-label={mode === "team" ? "团队面板" : "活动面板"}
      aria-modal={modal || undefined}
    >
      <header>
        <div>
          <span>{mode === "team" ? "COLLABORATION" : "EVENT STREAM"}</span>
          <h2>{mode === "team" ? "Agent Team" : "活动"}</h2>
        </div>
        <button
          type="button"
          className="shell-icon-button"
          onClick={onClose}
          aria-label="关闭检查器"
          title="关闭检查器"
        >
          <X size={16} aria-hidden="true" />
        </button>
      </header>
      <div className="inspector-scroll">
        {mode === "activity" ? (
          <ActivityTimeline events={current.activity} />
        ) : member ? (
          <MemberDetail member={member} />
        ) : current.team ? (
          <div className="inspector-members">
            {current.team.members.map((item) => (
              <button
                type="button"
                key={item.member_id}
                onClick={() => onMember(item)}
              >
                <span className="member-avatar" aria-hidden="true">
                  {item.is_coordinator ? "C" : item.role.slice(0, 1).toUpperCase()}
                </span>
                <span>
                  <strong>{humanize(item.role)}</strong>
                  <small>{item.status}</small>
                </span>
                <ChevronRight size={14} aria-hidden="true" />
              </button>
            ))}
          </div>
        ) : (
          <UnavailableWorkspace title="团队尚未建立" />
        )}
      </div>
    </aside>
  );
}

function MemberDetail({
  member,
}: {
  member: AgentShellMemberProjection;
}) {
  return (
    <div className="member-detail">
      <span className="member-avatar large" aria-hidden="true">
        {member.member.is_coordinator
          ? "C"
          : member.member.role.slice(0, 1).toUpperCase()}
      </span>
      <h3>{humanize(member.member.role)}</h3>
      <code>{shortIdentity(member.member.member_id, 38)}</code>
      <dl>
        <div>
          <dt>任务</dt>
          <dd>{member.task_refs.length}</dd>
        </div>
        <div>
          <dt>产物</dt>
          <dd>{member.artifact_envelope_refs.length}</dd>
        </div>
        <div>
          <dt>消息</dt>
          <dd>{member.messages.length}</dd>
        </div>
        <div>
          <dt>订阅</dt>
          <dd>{member.subscription_refs.length}</dd>
        </div>
      </dl>
      <section>
        <h4>消息元数据</h4>
        {member.messages.map((message) => (
          <article key={message.message_ref.object_id}>
            <strong>{humanize(message.message_kind)}</strong>
            <small>{humanize(message.audience)}</small>
          </article>
        ))}
        {!member.messages.length && <p>暂无消息</p>}
      </section>
    </div>
  );
}

function ShellStatusBar({
  connection,
  current,
  notice,
}: {
  connection: AgentShellConnectionState;
  current: AgentShellProjection | null;
  notice: string;
}) {
  return (
    <footer className="shell-status-bar">
      <span className={`stream-state ${connection}`}>
        {connection === "online" ? (
          <Wifi size={13} aria-hidden="true" />
        ) : connection === "offline" ? (
          <WifiOff size={13} aria-hidden="true" />
        ) : (
          <span className="shell-spinner small" aria-hidden="true" />
        )}
        {connectionLabel(connection)}
      </span>
      <span aria-live="polite">{notice}</span>
      <code>
        {current
          ? `seq ${current.reconnect_cursor} / v${current.session.session_version}`
          : "no session"}
      </code>
    </footer>
  );
}

function PhaseBadge({
  phase,
}: {
  phase: AgentShellProjection["execution_phase"];
}) {
  return (
    <span className={`phase-badge phase-${phase.toLowerCase()}`}>
      {phase === "COMPLETED" ? (
        <CheckCircle2 size={14} aria-hidden="true" />
      ) : (
        <Circle size={13} aria-hidden="true" />
      )}
      {phaseLabel(phase)}
    </span>
  );
}

function StatusDot({ status }: { status: string }) {
  return (
    <span
      className={`status-dot status-${status.toLowerCase()}`}
      title={humanize(status)}
      aria-label={humanize(status)}
    />
  );
}

function shellModeForWorkspace(
  workspace: AgentShellWorkspaceKind,
): ShellMode {
  if (workspace === "CONVERSATION") return "CONVERSATION";
  if (
    (RUN_WORKSPACES as readonly AgentShellWorkspaceKind[]).includes(
      workspace,
    )
  ) {
    return "RUN";
  }
  return "WORKBENCH";
}

function resolveModeWorkspace(
  mode: ShellMode,
  current: AgentShellProjection,
  workspace: AgentShellWorkspaceKind,
): AgentShellWorkspaceKind {
  if (mode === "CONVERSATION") return "CONVERSATION";
  const candidates: readonly AgentShellWorkspaceKind[] =
    mode === "RUN" ? RUN_WORKSPACES : WORKBENCH_WORKSPACES;
  if (candidates.includes(workspace)) {
    return workspace;
  }
  const descriptor = (kind: AgentShellWorkspaceKind) =>
    current.workspaces.find((item) => item.kind === kind);
  return (
    candidates.find((kind) => {
      const item = descriptor(kind);
      return item?.status === "AVAILABLE" || item?.status === "COMPLETE";
    }) ??
    candidates.find((kind) => descriptor(kind)?.status !== "BLOCKED") ??
    candidates[0]
  );
}

function modeItemCount(
  mode: ShellMode,
  workspaces: AgentShellWorkspaceDescriptor[],
): number {
  if (mode === "CONVERSATION") return 0;
  const kinds =
    mode === "RUN" ? RUN_WORKSPACES : WORKBENCH_WORKSPACES;
  return workspaces
    .filter((item) =>
      (kinds as readonly AgentShellWorkspaceKind[]).includes(
        item.kind,
      ),
    )
    .reduce((total, item) => total + item.item_count, 0);
}

function uniqueSourceFiles(files: FileList | null): File[] {
  const unique = new Map<string, File>();
  for (const file of Array.from(files ?? [])) {
    if (!unique.has(file.name)) unique.set(file.name, file);
  }
  return [...unique.values()];
}

function composeMessageContent(
  draft: string,
  quotes: ComposerQuote[],
): string {
  const context = quotes
    .map(
      (quote, index) =>
        `[引用 Agent 输出 ${index + 1}]\n${quote.content}`,
    )
    .join("\n\n");
  return [context, draft.trim()].filter(Boolean).join("\n\n");
}

function selectedMessageContent(
  event: ReactMouseEvent<HTMLButtonElement>,
  fallback: string,
): string {
  const selection = window.getSelection();
  const container = event.currentTarget.closest(".message-content");
  if (
    !selection ||
    selection.isCollapsed ||
    !selection.rangeCount ||
    !container?.contains(selection.getRangeAt(0).commonAncestorContainer)
  ) {
    return fallback;
  }
  return selection.toString().trim() || fallback;
}

function compactText(value: string, maxLength: number): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length > maxLength
    ? `${compact.slice(0, maxLength - 1)}…`
    : compact;
}

function upsertSession(
  sessions: AgentShellSessionSummary[],
  next: AgentShellSessionSummary,
): AgentShellSessionSummary[] {
  return [
    next,
    ...sessions.filter(
      (item) => item.session_id !== next.session_id,
    ),
  ].sort((left, right) =>
    right.updated_at.localeCompare(left.updated_at),
  );
}

function workspaceFromLocation(): {
  sessionId: string | null;
  workspace: AgentShellWorkspaceKind;
} {
  const segments = window.location.pathname.split("/").filter(Boolean);
  if (segments[0] !== "sessions" || !segments[1]) {
    return { sessionId: null, workspace: "CONVERSATION" };
  }
  const raw =
    segments[2] === "workspace" && segments[3]
      ? segments[3].toUpperCase()
      : segments[2] === "team"
        ? "TEAM"
        : "CONVERSATION";
  return {
    sessionId: decodeURIComponent(segments[1]),
    workspace: isWorkspaceKind(raw) ? raw : "CONVERSATION",
  };
}

function writeRoute(
  sessionId: string,
  workspace: AgentShellWorkspaceKind,
) {
  const suffix =
    workspace === "CONVERSATION"
      ? ""
      : `/workspace/${workspace.toLocaleLowerCase()}`;
  window.history.pushState(
    null,
    "",
    `/sessions/${encodeURIComponent(sessionId)}${suffix}`,
  );
}

function writeMemberRoute(sessionId: string, memberId: string) {
  window.history.pushState(
    null,
    "",
    `/sessions/${encodeURIComponent(sessionId)}/team/${encodeURIComponent(
      memberId,
    )}`,
  );
}

function isWorkspaceKind(
  value: string,
): value is AgentShellWorkspaceKind {
  return Object.hasOwn(WORKSPACE_LABELS, value);
}

function phaseLabel(phase: AgentShellExecutionPhase): string {
  const labels: Record<AgentShellExecutionPhase, string> = {
    NOT_CONFIGURED: "未配置",
    WAITING_REQUIREMENT: "等待需求",
    WAITING_SOURCE: "等待来源",
    READY: "已就绪",
    RUNNING: "执行中",
    WAITING_REVIEW: "等待审核",
    VERIFICATION_REQUIRED: "需要核验",
    BLOCKED: "已阻塞",
    COMPLETED: "已完成",
  };
  return labels[phase];
}

function interactionLabel(
  kind: AgentShellInteractionCard["kind"],
): string {
  return {
    CLARIFICATION: "需求澄清",
    PERMISSION: "权限确认",
    PLAN_REVIEW: "计划审核",
    VERIFICATION: "结果核验",
  }[kind];
}

function interactionTitle(
  kind: AgentShellInteractionCard["kind"],
): string {
  return {
    CLARIFICATION: "Agent 需要补充信息",
    PERMISSION: "执行需要明确授权",
    PLAN_REVIEW: "计划等待你的审核",
    VERIFICATION: "执行结果需要人工核验",
  }[kind];
}

function connectionLabel(
  state: AgentShellConnectionState,
): string {
  return {
    connecting: "正在连接",
    online: "实时同步",
    reconnecting: "正在重连",
    offline: "连接中断",
  }[state];
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () =>
      typeof window.matchMedia === "function" &&
      window.matchMedia(query).matches,
  );

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const media = window.matchMedia(query);
    const onChange = () => setMatches(media.matches);
    onChange();
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

function useOverlayFocus(
  open: boolean,
  selector: string,
  modal: boolean,
  onClose: () => void,
) {
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    const container = document.querySelector<HTMLElement>(selector);
    if (!container) return;
    const previous =
      modal && document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const focusable = () =>
      Array.from(
        container.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => element.getClientRects().length > 0);
    const frame = modal
      ? window.requestAnimationFrame(() => focusable()[0]?.focus())
      : null;

    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (!modal || event.key !== "Tab") return;
      const values = focusable();
      if (!values.length) {
        event.preventDefault();
        return;
      }
      const first = values[0];
      const last = values[values.length - 1];
      if (
        event.shiftKey &&
        (document.activeElement === first ||
          !container.contains(document.activeElement))
      ) {
        event.preventDefault();
        last.focus();
      } else if (
        !event.shiftKey &&
        (document.activeElement === last ||
          !container.contains(document.activeElement))
      ) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKeyDown);
      if (previous?.isConnected) previous.focus();
    };
  }, [modal, open, selector]);
}

function speechRecognitionConstructor(): BrowserSpeechRecognitionConstructor | null {
  const browser = window as typeof window & {
    SpeechRecognition?: BrowserSpeechRecognitionConstructor;
    webkitSpeechRecognition?: BrowserSpeechRecognitionConstructor;
  };
  return (
    browser.SpeechRecognition ??
    browser.webkitSpeechRecognition ??
    null
  );
}

function formatRelativeTime(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return value;
  const minutes = Math.max(
    0,
    Math.floor((Date.now() - timestamp) / 60_000),
  );
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)} 小时前`;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric",
    day: "numeric",
  }).format(new Date(timestamp));
}

function formatClock(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function shortIdentity(value: string, max = 32): string {
  if (value.length <= max) return value;
  const keep = Math.max(6, Math.floor((max - 3) / 2));
  return `${value.slice(0, keep)}...${value.slice(-keep)}`;
}

function humanize(value: string): string {
  return value
    .replaceAll("_", " ")
    .toLocaleLowerCase()
    .replace(/\b\w/g, (letter) => letter.toLocaleUpperCase());
}

function errorMessage(reason: unknown): string {
  if (reason instanceof ApiError) {
    return `${reason.code}: ${reason.message}`;
  }
  if (reason instanceof Error) return reason.message;
  return "Agent Shell 发生未知错误。";
}
