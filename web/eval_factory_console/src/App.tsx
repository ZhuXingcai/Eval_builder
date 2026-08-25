import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  Activity,
  AlertTriangle,
  Archive,
  Check,
  ChevronDown,
  ChevronRight,
  CircleDot,
  Database,
  Download,
  FileJson2,
  FileText,
  FolderTree,
  GitBranch,
  Inbox,
  Layers3,
  Menu,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  X,
  XCircle,
  type LucideIcon,
} from "lucide-react";

import {
  ApiError,
  decidePlanReview,
  editPlanReview,
  fetchApiContract,
  listPlanReviews,
  resumePlanReview,
} from "./api";
import {
  editableFields,
  isAttachmentGenerationPlan,
  isCriteriaRubricPlan,
  isDatasetBuildPlan,
  isDatasetDeliveryPlan,
  isGradingDesignPlan,
  type GradingDesignPlan,
  type PlanEditableFields,
  type PlanReviewApiContract,
  type PlanReviewState,
  type PlanReviewView,
  type ReviewablePlan,
} from "./types";

const DEFAULT_PRINCIPAL = "user://plan-owner";

export interface PlanReviewWorkspaceProps {
  embedded?: boolean;
  initialReviewId?: string | null;
  onReviewChange?: (view: PlanReviewView) => void;
  principal?: string;
}

type ActiveView = "overview" | "json";

type EditorState = {
  reviewId: string | null;
  planVersion: number | null;
  value: string;
};

type ReviewGroup = {
  runId: string;
  reviews: PlanReviewView[];
};

type PlanFact = {
  label: string;
  value: string;
};

const STAGES = [
  "规划",
  "轨迹",
  "任务",
  "附件",
  "准则",
  "评分",
  "交付",
] as const;

const NAV_ITEMS: ReadonlyArray<{
  label: string;
  Icon: LucideIcon;
  active?: boolean;
}> = [
  { label: "审核", Icon: Inbox, active: true },
  { label: "轨迹", Icon: GitBranch },
  { label: "运行", Icon: Database },
  { label: "产物", Icon: Archive },
];

export default function App({
  embedded = false,
  initialReviewId = null,
  onReviewChange,
  principal = DEFAULT_PRINCIPAL,
}: PlanReviewWorkspaceProps = {}) {
  const [contract, setContract] =
    useState<PlanReviewApiContract | null>(null);
  const [reviews, setReviews] = useState<PlanReviewView[]>([]);
  const [selected, setSelected] =
    useState<PlanReviewView | null>(null);
  const selectedRef = useRef<PlanReviewView | null>(null);
  const [editor, setEditor] = useState<EditorState>(
    emptyEditor(),
  );
  const [notice, setNotice] = useState(
    "正在连接控制平面",
  );
  const [activeOperation, setActiveOperation] = useState<
    string | null
  >(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [activeView, setActiveView] =
    useState<ActiveView>("overview");
  const [activityOpen, setActivityOpen] = useState(false);
  const [mobileNavigationOpen, setMobileNavigationOpen] =
    useState(false);

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!activityOpen && !mobileNavigationOpen) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setActivityOpen(false);
      setMobileNavigationOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activityOpen, mobileNavigationOpen]);

  const filteredReviews = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    if (!normalized) return reviews;
    return reviews.filter((review) =>
      [
        review.presentation.title,
        review.run_id,
        review.request.plan_kind,
        review.result.state,
        review.request.review_request_id,
      ]
        .join(" ")
        .toLocaleLowerCase()
        .includes(normalized),
    );
  }, [query, reviews]);

  const reviewGroups = useMemo(
    () => groupReviews(filteredReviews),
    [filteredReviews],
  );

  const changedPaths = useMemo(() => {
    if (
      !selected ||
      editor.reviewId !== selected.request.review_request_id ||
      editor.planVersion !== selected.request.plan_version
    ) {
      return [];
    }
    try {
      const next = JSON.parse(editor.value) as Record<
        string,
        unknown
      >;
      const before = editableFields(
        selected.plan,
      ) as unknown as Record<string, unknown>;
      return Object.keys(before)
        .filter(
          (key) =>
            JSON.stringify(before[key]) !==
            JSON.stringify(next[key]),
        )
        .sort();
    } catch {
      return [];
    }
  }, [editor, selected]);

  const busy = activeOperation !== null;
  const state = selected?.result.state ?? "NO_SELECTION";
  const canDecide = state === "PENDING_REVIEW";
  const canResume =
    state === "APPROVED" || state === "REVISION_REQUESTED";

  function selectReview(
    review: PlanReviewView | null,
    options?: { preserveView?: boolean },
  ) {
    selectedRef.current = review;
    setSelected(review);
    setEditor(editorFor(review));
    if (!options?.preserveView) setActiveView("overview");
    setActivityOpen(false);
    setMobileNavigationOpen(false);
    setError(null);
  }

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const [apiContract, page] = await Promise.all([
        fetchApiContract(),
        listPlanReviews("PENDING_REVIEW"),
      ]);
      setContract(apiContract);
      setReviews(page.items);

      const currentId =
        selectedRef.current?.request.review_request_id ??
        initialReviewId;
      const nextSelected =
        page.items.find(
          (item) =>
            item.request.review_request_id === currentId,
        ) ??
        page.items[0] ??
        null;
      selectReview(nextSelected, { preserveView: true });
      setNotice(
        page.total
          ? `${page.total} 个计划等待审核`
          : "暂无待审核计划",
      );
    } catch (reason) {
      setError(message(reason));
      setNotice("控制平面不可用");
    } finally {
      setLoading(false);
    }
  }

  async function act(
    operation: (view: PlanReviewView) => Promise<PlanReviewView>,
    operationLabel: string,
    success: string,
  ) {
    if (!selected) return;
    setActiveOperation(operationLabel);
    setError(null);
    try {
      const next = await operation(selected);
      selectReview(next, { preserveView: true });
      setReviews((values) =>
        values.map((value) =>
          value.request.review_request_id ===
          next.request.review_request_id
            ? next
            : value,
        ),
      );
      setNotice(success);
      onReviewChange?.(next);
    } catch (reason) {
      setError(message(reason));
      setActivityOpen(true);
    } finally {
      setActiveOperation(null);
    }
  }

  async function saveEdit() {
    if (
      !selected ||
      editor.reviewId !== selected.request.review_request_id ||
      editor.planVersion !== selected.request.plan_version
    ) {
      setError(
        "编辑器仍在同步当前所选计划。",
      );
      return;
    }
    let parsed: PlanEditableFields;
    try {
      parsed = JSON.parse(editor.value) as PlanEditableFields;
    } catch {
      setError("修改后的计划不是有效的 JSON。");
      return;
    }
    if (!changedPaths.length) {
      setError("至少需要修改一个计划字段。");
      return;
    }
    await act(
      (view) =>
        editPlanReview(
          view,
          parsed,
          changedPaths,
          principal,
        ),
      "正在提交修改",
      "后继计划已编译并记录",
    );
  }

  function downloadPlan() {
    if (!selected) return;
    const blob = new Blob([JSON.stringify(selected, null, 2)], {
      type: "application/json",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `plan-review-${selected.request.plan_version}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  const approve = () =>
    act(
      (view) =>
        decidePlanReview(
          view,
          "APPROVE",
          principal,
          "APPROVE_BY_WEB",
        ),
      "正在批准计划",
      "计划已批准",
    );
  const requestMore = () =>
    act(
      (view) =>
        decidePlanReview(
          view,
          "REQUEST_MORE",
          principal,
          "REQUEST_MORE_BY_WEB",
        ),
      "正在请求补充材料",
      "已请求补充规划材料",
    );
  const defer = () =>
    act(
      (view) =>
        decidePlanReview(
          view,
          "DEFER",
          principal,
          "DEFER_BY_WEB",
        ),
      "正在暂缓计划",
      "计划已暂缓",
    );
  const reject = () =>
    act(
      (view) =>
        decidePlanReview(
          view,
          "REJECT",
          principal,
          "REJECT_BY_WEB",
        ),
      "正在拒绝计划",
      "计划已拒绝",
    );
  const resume = () =>
    act(
      (view) => resumePlanReview(view, principal),
      "正在恢复图执行",
      "图恢复权限已提交",
    );

  return (
    <div
      className={[
        "workbench-shell",
        activityOpen ? "has-activity" : "",
        embedded ? "embedded-plan-review" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <a className="skip-link" href="#plan-workspace">
        跳转到当前审核
      </a>

      {(mobileNavigationOpen || activityOpen) && (
        <button
          type="button"
          className="panel-backdrop"
          aria-label="关闭当前面板"
          onClick={() => {
            setMobileNavigationOpen(false);
            setActivityOpen(false);
          }}
        />
      )}

      <ReviewNavigation
        contract={contract}
        groups={reviewGroups}
        loading={loading}
        mobileOpen={mobileNavigationOpen}
        onClose={() => setMobileNavigationOpen(false)}
        onRefresh={() => void refresh()}
        onSelect={selectReview}
        onSearch={setQuery}
        query={query}
        selectedId={selected?.request.review_request_id ?? null}
        total={reviews.length}
        principal={principal}
      />

      <main className="workbench-main">
        <WorkspaceTabs
          activeView={activeView}
          activityCount={
            (selected?.presentation.warning_codes.length ?? 0) +
            changedPaths.length
          }
          activityOpen={activityOpen}
          onActivityToggle={() => {
            setActivityOpen((current) => !current);
            setMobileNavigationOpen(false);
          }}
          onMobileNavigation={() => {
            setMobileNavigationOpen(true);
            setActivityOpen(false);
          }}
          onViewChange={setActiveView}
          selected={selected}
        />

        {error && (
          <div className="error-banner" role="alert">
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
          id="plan-workspace"
          className="plan-workspace"
          aria-label="当前计划工作区"
        >
          {selected ? (
            <>
              <WorkspaceHeader selected={selected} />
              <StageStrip planKind={selected.request.plan_kind} />
              {activeView === "overview" ? (
                <OverviewPanel selected={selected} />
              ) : (
                <JsonWorkspace
                  busy={busy}
                  canDecide={canDecide}
                  changedPaths={changedPaths}
                  editor={editor}
                  onChange={(value) =>
                    setEditor((current) => ({
                      ...current,
                      value,
                    }))
                  }
                  onSave={() => void saveEdit()}
                  selected={selected}
                />
              )}
            </>
          ) : (
            <NoSelection loading={loading} error={error} />
          )}
        </section>

        {selected && activityOpen && (
          <ReviewActivity
            busy={busy}
            canDecide={canDecide}
            changedPaths={changedPaths}
            onClose={() => setActivityOpen(false)}
            onCommitEdit={() => void saveEdit()}
            onDefer={() => void defer()}
            onExport={downloadPlan}
            onRefresh={() => void refresh()}
            onReject={() => void reject()}
            onRequestMore={() => void requestMore()}
            selected={selected}
          />
        )}

        <AuthorityBar
          activeOperation={activeOperation}
          canDecide={canDecide}
          canResume={canResume}
          changedCount={changedPaths.length}
          notice={notice}
          onApprove={() => void approve()}
          onResume={() => void resume()}
          selected={selected}
          principal={principal}
        />
      </main>
    </div>
  );
}

function ReviewNavigation({
  contract,
  groups,
  loading,
  mobileOpen,
  onClose,
  onRefresh,
  onSearch,
  onSelect,
  query,
  selectedId,
  total,
  principal,
}: {
  contract: PlanReviewApiContract | null;
  groups: ReviewGroup[];
  loading: boolean;
  mobileOpen: boolean;
  onClose: () => void;
  onRefresh: () => void;
  onSearch: (value: string) => void;
  onSelect: (review: PlanReviewView) => void;
  query: string;
  selectedId: string | null;
  total: number;
  principal: string;
}) {
  return (
    <aside
      className={`review-navigation ${mobileOpen ? "mobile-open" : ""}`}
      aria-label="计划审核导航"
    >
      <div className="navigation-brand">
        <span className="brand-mark" aria-hidden="true">
          数
        </span>
        <div>
          <strong>评测数据集工厂</strong>
          <span>
            <CircleDot size={10} aria-hidden="true" />
            {contract ? "控制平面在线" : "正在同步"}
          </span>
        </div>
        <button
          type="button"
          className="icon-button mobile-only"
          aria-label="关闭审核导航"
          title="关闭审核导航"
          onClick={onClose}
        >
          <X size={17} aria-hidden="true" />
        </button>
      </div>

      <nav className="product-navigation" aria-label="产品导航">
        {NAV_ITEMS.map(({ label, Icon, active }) => (
          <button
            type="button"
            key={label}
            className={active ? "active" : ""}
            disabled={!active}
            aria-current={active ? "page" : undefined}
            title={active ? label : `${label}尚未接入`}
          >
            <Icon size={16} aria-hidden="true" />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className="queue-heading">
        <div>
          <span>审核队列</span>
          <strong>{total}</strong>
        </div>
        <button
          type="button"
          className="icon-button dark"
          onClick={onRefresh}
          disabled={loading}
          aria-label="刷新审核队列"
          title="刷新审核队列"
        >
          <RefreshCw
            size={15}
            className={loading ? "spin" : ""}
            aria-hidden="true"
          />
        </button>
      </div>

      <label className="review-search">
        <Search size={15} aria-hidden="true" />
        <span className="sr-only">搜索审核项</span>
        <input
          type="search"
          value={query}
          onChange={(event) => onSearch(event.target.value)}
          placeholder="搜索审核项"
        />
        {query && (
          <button
            type="button"
            aria-label="清空审核搜索"
            title="清空审核搜索"
            onClick={() => onSearch("")}
          >
            <X size={13} aria-hidden="true" />
          </button>
        )}
      </label>

      <div className="review-tree" aria-label="计划审核队列">
        {groups.map((group) => (
          <section className="run-group" key={group.runId}>
            <div className="run-heading" title={group.runId}>
              <ChevronDown size={13} aria-hidden="true" />
              <FolderTree size={14} aria-hidden="true" />
              <span>{shortIdentity(group.runId)}</span>
              <small>{group.reviews.length}</small>
            </div>
            <div className="run-reviews">
              {group.reviews.map((review) => {
                const active =
                  review.request.review_request_id === selectedId;
                return (
                  <button
                    type="button"
                    data-review-id={
                      review.request.review_request_id
                    }
                    className={`review-row ${active ? "active" : ""}`}
                    key={review.request.review_request_id}
                    onClick={() => onSelect(review)}
                    aria-current={active ? "true" : undefined}
                  >
                    <StateGlyph state={review.result.state} />
                    <span className="review-row-copy">
                      <strong>{review.presentation.title}</strong>
                      <small>
                        {humanize(review.request.plan_kind)}
                        <span aria-hidden="true"> / </span>v
                        {review.request.plan_version}
                      </small>
                    </span>
                    <ChevronRight
                      size={14}
                      className="row-chevron"
                      aria-hidden="true"
                    />
                  </button>
                );
              })}
            </div>
          </section>
        ))}
        {!groups.length && (
          <div className="queue-empty" role="status">
            <Inbox size={19} aria-hidden="true" />
            <span>
              {loading
                ? "正在加载审核项"
                : query
                  ? "没有匹配的审核项"
                  : "暂无待审核计划"}
            </span>
          </div>
        )}
      </div>

      <footer className="navigation-footer">
        <ShieldCheck size={15} aria-hidden="true" />
        <span>
          <small>当前主体</small>
          <code>{principal}</code>
        </span>
      </footer>
    </aside>
  );
}

function WorkspaceTabs({
  activeView,
  activityCount,
  activityOpen,
  onActivityToggle,
  onMobileNavigation,
  onViewChange,
  selected,
}: {
  activeView: ActiveView;
  activityCount: number;
  activityOpen: boolean;
  onActivityToggle: () => void;
  onMobileNavigation: () => void;
  onViewChange: (view: ActiveView) => void;
  selected: PlanReviewView | null;
}) {
  return (
    <header className="workspace-tabs">
      <button
        type="button"
        className="icon-button mobile-only"
        aria-label="打开审核导航"
        title="打开审核导航"
        onClick={onMobileNavigation}
      >
        <Menu size={18} aria-hidden="true" />
      </button>

      <div className="workspace-identity">
        <Layers3 size={16} aria-hidden="true" />
        <span title={selected?.presentation.title}>
          {selected?.presentation.title ?? "审核工作区"}
        </span>
      </div>

      <div
        className="view-tabs"
        role="tablist"
        aria-label="计划工作区视图"
      >
        <button
          type="button"
          role="tab"
          aria-label="概览"
          aria-selected={activeView === "overview"}
          className={activeView === "overview" ? "active" : ""}
          onClick={() => onViewChange("overview")}
        >
          <FileText size={14} aria-hidden="true" />
          <span>概览</span>
        </button>
        <button
          type="button"
          role="tab"
          aria-label="计划 JSON"
          aria-selected={activeView === "json"}
          className={activeView === "json" ? "active" : ""}
          onClick={() => onViewChange("json")}
          disabled={!selected}
        >
          <FileJson2 size={14} aria-hidden="true" />
          <span>计划 JSON</span>
        </button>
      </div>

      <button
        type="button"
        className={`activity-toggle ${activityOpen ? "active" : ""}`}
        aria-expanded={activityOpen}
        aria-controls="review-activity"
        aria-label="审核动态"
        onClick={onActivityToggle}
        disabled={!selected}
        title="审核动态"
      >
        <Activity size={15} aria-hidden="true" />
        <span>动态</span>
        {activityCount > 0 && <strong>{activityCount}</strong>}
      </button>
    </header>
  );
}

function WorkspaceHeader({
  selected,
}: {
  selected: PlanReviewView;
}) {
  return (
    <header className="workspace-header">
      <div className="workspace-breadcrumb">
        <span>{shortIdentity(selected.run_id)}</span>
        <ChevronRight size={12} aria-hidden="true" />
        <span>{humanize(selected.request.plan_kind)}</span>
      </div>
      <div className="workspace-title-row">
        <div>
          <h1>{selected.presentation.title}</h1>
          <p>
            请求主体{" "}
            <code>{selected.request.requested_by}</code>
          </p>
        </div>
        <StateBadge state={selected.result.state} />
      </div>
      <dl className="workspace-metadata">
        <div>
          <dt>运行 ID</dt>
          <dd title={selected.run_id}>
            {shortIdentity(selected.run_id, 34)}
          </dd>
        </div>
        <div>
          <dt>计划类型</dt>
          <dd>{humanize(selected.request.plan_kind)}</dd>
        </div>
        <div>
          <dt>版本</dt>
          <dd>v{selected.request.plan_version}</dd>
        </div>
        <div>
          <dt>权限哈希</dt>
          <dd
            data-authority-hash={
              selected.request.plan_ref.object_sha256
            }
            title={selected.request.plan_ref.object_sha256}
          >
            {selected.request.plan_ref.object_sha256.slice(0, 12)}
          </dd>
        </div>
      </dl>
    </header>
  );
}

function StageStrip({ planKind }: { planKind: string }) {
  const currentIndex = stageIndex(planKind);
  return (
    <ol
      className="stage-strip"
      aria-label="计划阶段位置"
    >
      {STAGES.map((stage, index) => (
        <li
          key={stage}
          className={index === currentIndex ? "current" : ""}
          aria-current={
            index === currentIndex ? "step" : undefined
          }
        >
          <span>{String(index + 1).padStart(2, "0")}</span>
          <strong>{stage}</strong>
        </li>
      ))}
    </ol>
  );
}

function OverviewPanel({
  selected,
}: {
  selected: PlanReviewView;
}) {
  return (
    <div className="overview-panel">
      <section className="review-brief" aria-labelledby="brief-title">
        <div className="section-heading">
          <div>
            <span>审核摘要</span>
            <h2 id="brief-title">权限概览</h2>
          </div>
          <small>
            {selected.presentation.summary_lines.length} 条信息
          </small>
        </div>
        <div className="summary-lines">
          {selected.presentation.summary_lines.map((line, index) => (
            <div key={`${index}-${line}`}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <p>{line}</p>
            </div>
          ))}
        </div>
      </section>

      <PlanFacts plan={selected.plan} />
      <PlanLedger plan={selected.plan} />
    </div>
  );
}

function PlanFacts({ plan }: { plan: ReviewablePlan }) {
  const facts = planFacts(plan);
  return (
    <dl className="facts-strip" aria-label="计划信息">
      {facts.map((fact) => (
        <div key={fact.label}>
          <dt>{fact.label}</dt>
          <dd>{fact.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function PlanLedger({ plan }: { plan: ReviewablePlan }) {
  const ledgerCount = isDatasetBuildPlan(plan)
    ? `${plan.tasks.length} 个任务`
    : isAttachmentGenerationPlan(plan)
      ? `${plan.works.length} 个分组`
      : isCriteriaRubricPlan(plan)
        ? `${plan.criterion_goals.length} 条准则`
        : isGradingDesignPlan(plan)
          ? `${plan.judge_tasks.length} 个评分任务`
          : `${plan.candidate_item_refs.length} 个候选项`;

  return (
    <section className="task-ledger" aria-labelledby="ledger-title">
      <div className="section-heading compact">
        <div>
          <span>计划结构</span>
          <h2 id="ledger-title">执行清单</h2>
        </div>
        <code>{ledgerCount}</code>
      </div>
      <div className="ledger-rows">
        {isDatasetBuildPlan(plan) &&
          plan.tasks.map((task) => (
            <LedgerRow
              key={task.task_key}
              identity={task.task_key}
              title={humanize(task.task_kind)}
              detail={`${task.agent_role}${
                task.dependency_task_keys.length
                  ? ` / 依赖 ${task.dependency_task_keys.join(", ")}`
                  : " / 独立执行"
              }`}
              meta={[
                `${task.max_attempts} 次尝试`,
                `${task.max_model_tokens.toLocaleString()} Token`,
              ]}
            />
          ))}
        {isAttachmentGenerationPlan(plan) &&
          plan.works.map((work) => (
            <LedgerRow
              key={work.work_key}
              identity={work.work_key}
              title={`${work.artifact_ids.length} 个产物`}
              detail={
                work.dependency_work_keys.length
                  ? `依赖 ${work.dependency_work_keys.join(", ")}`
                  : "独立分组"
              }
              meta={[
                `${work.max_attempts} 次尝试`,
                `${work.max_model_tokens.toLocaleString()} Token`,
              ]}
            />
          ))}
        {isCriteriaRubricPlan(plan) &&
          plan.criterion_goals.map((goal) => (
            <LedgerRow
              key={goal.goal_id}
              identity={`${(
                goal.weight_basis_points / 100
              ).toFixed(0)}%`}
              title={humanize(goal.judged_object_kind)}
              detail={goal.goal_summary}
              meta={[
                goal.evaluator_binding_id,
                humanize(goal.visibility),
              ]}
            />
          ))}
        {isGradingDesignPlan(plan) && (
          <GradingLedger plan={plan} />
        )}
        {isDatasetDeliveryPlan(plan) && (
          <LedgerRow
            identity={String(plan.candidate_item_refs.length)}
            title="候选项交付"
            detail="生产发布已禁用"
            meta={[
              `${plan.max_files.toLocaleString()} 个文件`,
              `${plan.max_total_bytes.toLocaleString()} 字节`,
            ]}
          />
        )}
      </div>
    </section>
  );
}

function LedgerRow({
  detail,
  identity,
  meta,
  title,
}: {
  detail: string;
  identity: string;
  meta: string[];
  title: string;
}) {
  return (
    <article className="task-row">
      <code className="task-key">{identity}</code>
      <div className="task-copy">
        <strong>{title}</strong>
        <small>{detail}</small>
      </div>
      <div className="task-budget">
        {meta.map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>
    </article>
  );
}

function GradingLedger({ plan }: { plan: GradingDesignPlan }) {
  return plan.judge_tasks.map((task) => (
    <LedgerRow
      key={task.task_key}
      identity={`${(
        task.score_weight_basis_points / 100
      ).toFixed(0)}%`}
      title={task.task_key}
      detail={`绑定 ${task.criterion_ids.length} 条准则`}
      meta={[task.evaluator_binding_id, plan.aggregation_mode]}
    />
  ));
}

function JsonWorkspace({
  busy,
  canDecide,
  changedPaths,
  editor,
  onChange,
  onSave,
  selected,
}: {
  busy: boolean;
  canDecide: boolean;
  changedPaths: string[];
  editor: EditorState;
  onChange: (value: string) => void;
  onSave: () => void;
  selected: PlanReviewView;
}) {
  const synchronized =
    editor.reviewId === selected.request.review_request_id &&
    editor.planVersion === selected.request.plan_version;
  return (
    <section className="json-workspace" aria-labelledby="json-title">
      <header className="json-toolbar">
        <div>
          <span>安全可编辑投影</span>
          <h2 id="json-title">计划 JSON</h2>
        </div>
        <div className="json-toolbar-actions">
          <span className="change-count">
            已修改 {changedPaths.length} 项
          </span>
          <button
            type="button"
            className="secondary-button"
            disabled={
              !canDecide ||
              busy ||
              !synchronized ||
              changedPaths.length === 0
            }
            onClick={onSave}
          >
            <Check size={15} aria-hidden="true" />
            提交修改
          </button>
        </div>
      </header>
      <textarea
        aria-label="可编辑计划 JSON"
        data-editor-review-id={editor.reviewId ?? ""}
        data-editor-plan-version={editor.planVersion ?? ""}
        spellCheck={false}
        value={editor.value}
        onChange={(event) => onChange(event.target.value)}
        disabled={!canDecide || busy || !synchronized}
      />
      <footer className="changed-paths">
        <span>修改字段</span>
        <div>
          {changedPaths.length ? (
            changedPaths.map((path) => (
              <code key={path}>{path}</code>
            ))
          ) : (
            <small>暂无待提交修改</small>
          )}
        </div>
      </footer>
    </section>
  );
}

function ReviewActivity({
  busy,
  canDecide,
  changedPaths,
  onClose,
  onCommitEdit,
  onDefer,
  onExport,
  onRefresh,
  onReject,
  onRequestMore,
  selected,
}: {
  busy: boolean;
  canDecide: boolean;
  changedPaths: string[];
  onClose: () => void;
  onCommitEdit: () => void;
  onDefer: () => void;
  onExport: () => void;
  onRefresh: () => void;
  onReject: () => void;
  onRequestMore: () => void;
  selected: PlanReviewView;
}) {
  return (
    <aside
      id="review-activity"
      className="review-activity action-rail"
      role="dialog"
      aria-label="审核动态"
      aria-modal="false"
    >
      <header>
        <div>
          <span>当前权限</span>
          <h2>审核动态</h2>
        </div>
        <button
          type="button"
          className="icon-button"
          aria-label="关闭审核动态"
          title="关闭审核动态"
          onClick={onClose}
        >
          <X size={16} aria-hidden="true" />
        </button>
      </header>

      <div className="activity-scroll">
        <section className="state-panel">
          <span>当前状态</span>
          <strong>{humanize(selected.result.state)}</strong>
          <small>计划版本 {selected.result.plan_version}</small>
        </section>

        <ActivitySection
          title="警告"
          count={selected.presentation.warning_codes.length}
        >
          <div className="warning-stack">
            {selected.presentation.warning_codes.length ? (
              selected.presentation.warning_codes.map((warning) => (
                <span key={warning}>
                  <AlertTriangle size={13} aria-hidden="true" />
                  {humanize(warning)}
                </span>
              ))
            ) : (
              <p className="activity-empty">暂无警告</p>
            )}
          </div>
        </ActivitySection>

        <ActivitySection title="待提交修改" count={changedPaths.length}>
          <div className="activity-paths">
            {changedPaths.length ? (
              changedPaths.map((path) => (
                <code key={path}>{path}</code>
              ))
            ) : (
              <p className="activity-empty">暂无待提交修改</p>
            )}
          </div>
        </ActivitySection>

        {(selected.decision || selected.revision) && (
          <ActivitySection title="已记录权限">
            <dl className="activity-details">
              {selected.decision && (
                <>
                  <div>
                    <dt>决策</dt>
                    <dd>
                      {humanize(selected.decision.decision)}
                    </dd>
                  </div>
                  <div>
                    <dt>原因代码</dt>
                    <dd>{selected.decision.reason_code}</dd>
                  </div>
                  <div>
                    <dt>决策主体</dt>
                    <dd>{selected.decision.decided_by}</dd>
                  </div>
                </>
              )}
              {selected.revision && (
                <div>
                  <dt>修订字段</dt>
                  <dd>
                    {selected.revision.changed_paths.join(", ")}
                  </dd>
                </div>
              )}
            </dl>
          </ActivitySection>
        )}

        <ActivitySection title="审核操作">
          <div className="action-stack">
            <button
              type="button"
              disabled={
                !canDecide || busy || changedPaths.length === 0
              }
              onClick={onCommitEdit}
            >
              <Check size={15} aria-hidden="true" />
              提交修改
            </button>
            <button
              type="button"
              disabled={!canDecide || busy}
              onClick={onRequestMore}
            >
              <FileText size={15} aria-hidden="true" />
              补充材料
            </button>
            <button
              type="button"
              disabled={!canDecide || busy}
              onClick={onDefer}
            >
              <Archive size={15} aria-hidden="true" />
              暂缓
            </button>
            <button
              type="button"
              className="danger-action"
              disabled={!canDecide || busy}
              onClick={onReject}
            >
              <XCircle size={15} aria-hidden="true" />
              拒绝
            </button>
          </div>
        </ActivitySection>

      </div>

      <footer>
        <div className="activity-footer-actions">
          <button
            type="button"
            className="icon-button"
            onClick={onRefresh}
            disabled={busy}
            aria-label="刷新审核队列"
            title="刷新审核队列"
          >
            <RefreshCw size={15} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={onExport}
            aria-label="导出 JSON"
            title="导出 JSON"
          >
            <Download size={15} aria-hidden="true" />
          </button>
        </div>
        <span>
          <small>权限哈希</small>
          <code>
            {selected.request.plan_ref.object_sha256.slice(0, 16)}
          </code>
        </span>
      </footer>
    </aside>
  );
}

function ActivitySection({
  children,
  count,
  title,
}: {
  children: ReactNode;
  count?: number;
  title: string;
}) {
  return (
    <section className="activity-section">
      <header>
        <h3>{title}</h3>
        {typeof count === "number" && <span>{count}</span>}
      </header>
      {children}
    </section>
  );
}

function AuthorityBar({
  activeOperation,
  canDecide,
  canResume,
  changedCount,
  notice,
  onApprove,
  onResume,
  selected,
  principal,
}: {
  activeOperation: string | null;
  canDecide: boolean;
  canResume: boolean;
  changedCount: number;
  notice: string;
  onApprove: () => void;
  onResume: () => void;
  selected: PlanReviewView | null;
  principal: string;
}) {
  return (
    <footer className="authority-bar">
      <div className="authority-status" aria-live="polite">
        <span
          className={`connection-dot ${
            notice === "控制平面不可用"
              ? "offline"
              : ""
          }`}
          aria-hidden="true"
        />
        <span>{activeOperation ?? notice}</span>
      </div>
      <dl>
        <div>
          <dt>当前主体</dt>
          <dd>{principal}</dd>
        </div>
        <div>
          <dt>版本</dt>
          <dd>
            {selected ? `v${selected.request.plan_version}` : "-"}
          </dd>
        </div>
        <div>
          <dt>修改项</dt>
          <dd>{changedCount}</dd>
        </div>
      </dl>
      {canDecide ? (
        <button
          type="button"
          className="primary-action"
          disabled={activeOperation !== null}
          onClick={onApprove}
        >
          <Check size={16} aria-hidden="true" />
          批准
        </button>
      ) : canResume ? (
        <button
          type="button"
          className="primary-action"
          disabled={activeOperation !== null}
          onClick={onResume}
        >
          <Play size={15} aria-hidden="true" />
          恢复图执行
        </button>
      ) : (
        <button type="button" className="primary-action" disabled>
          <ShieldCheck size={15} aria-hidden="true" />
          暂无操作
        </button>
      )}
    </footer>
  );
}

function NoSelection({
  error,
  loading,
}: {
  error: string | null;
  loading: boolean;
}) {
  return (
    <div className="no-selection" role="status">
      {error ? (
        <AlertTriangle size={22} aria-hidden="true" />
      ) : loading ? (
        <RefreshCw className="spin" size={22} aria-hidden="true" />
      ) : (
        <Inbox size={22} aria-hidden="true" />
      )}
      <h1>
        {error
          ? "控制平面不可用"
          : loading
            ? "正在加载审核权限"
            : "暂无待审核计划"}
      </h1>
    </div>
  );
}

function StateGlyph({ state }: { state: PlanReviewState }) {
  if (state === "APPROVED" || state === "RESUMED") {
    return (
      <span className="state-glyph success" title={humanize(state)}>
        <Check size={11} aria-hidden="true" />
      </span>
    );
  }
  if (state === "REJECTED") {
    return (
      <span className="state-glyph danger" title={humanize(state)}>
        <X size={11} aria-hidden="true" />
      </span>
    );
  }
  return (
    <span className="state-glyph pending" title={humanize(state)}>
      <span aria-hidden="true" />
    </span>
  );
}

function StateBadge({ state }: { state: PlanReviewState }) {
  return (
    <span className={`state-badge state-${state.toLowerCase()}`}>
      <StateGlyph state={state} />
      {humanize(state)}
    </span>
  );
}

function editorFor(review: PlanReviewView | null): EditorState {
  if (!review) return emptyEditor();
  return {
    reviewId: review.request.review_request_id,
    planVersion: review.request.plan_version,
    value: JSON.stringify(editableFields(review.plan), null, 2),
  };
}

function emptyEditor(): EditorState {
  return {
    reviewId: null,
    planVersion: null,
    value: "",
  };
}

function groupReviews(reviews: PlanReviewView[]): ReviewGroup[] {
  const groups = new Map<string, PlanReviewView[]>();
  for (const review of reviews) {
    const existing = groups.get(review.run_id);
    if (existing) existing.push(review);
    else groups.set(review.run_id, [review]);
  }
  return Array.from(groups, ([runId, values]) => ({
    runId,
    reviews: values,
  }));
}

function planFacts(plan: ReviewablePlan): PlanFact[] {
  if (isDatasetBuildPlan(plan)) {
    return [
      { label: "任务数", value: plan.tasks.length.toLocaleString() },
      {
        label: "请求预算",
        value: plan.total_model_requests.toLocaleString(),
      },
      {
        label: "Token 预算",
        value: plan.total_model_tokens.toLocaleString(),
      },
      {
        label: "成本上限",
        value: `${plan.total_cost_micro_usd.toLocaleString()} 微美元`,
      },
    ];
  }
  if (isAttachmentGenerationPlan(plan)) {
    return [
      { label: "分组数", value: plan.works.length.toLocaleString() },
      {
        label: "并行上限",
        value: plan.max_parallel_groups.toLocaleString(),
      },
      {
        label: "Token 预算",
        value: plan.total_model_tokens.toLocaleString(),
      },
      {
        label: "成本上限",
        value: `${plan.total_cost_micro_usd.toLocaleString()} 微美元`,
      },
    ];
  }
  if (isCriteriaRubricPlan(plan)) {
    return [
      {
        label: "准则数",
        value: plan.criterion_goals.length.toLocaleString(),
      },
      {
        label: "参考模式",
        value: humanize(plan.selected_reference_mode),
      },
      {
        label: "Token 上限",
        value: plan.max_model_tokens.toLocaleString(),
      },
      {
        label: "尝试次数",
        value: plan.max_attempts.toLocaleString(),
      },
    ];
  }
  if (isGradingDesignPlan(plan)) {
    return [
      {
        label: "评分任务",
        value: plan.judge_tasks.length.toLocaleString(),
      },
      {
        label: "通过分数",
        value: `${(plan.passing_score_basis_points / 100).toFixed(0)}%`,
      },
      {
        label: "置信度下限",
        value: `${(
          plan.minimum_confidence_basis_points / 100
        ).toFixed(0)}%`,
      },
      {
        label: "聚合方式",
        value: humanize(plan.aggregation_mode),
      },
    ];
  }
  return [
    {
      label: "候选项",
      value: plan.candidate_item_refs.length.toLocaleString(),
    },
    {
      label: "文件上限",
      value: plan.max_files.toLocaleString(),
    },
    {
      label: "字节上限",
      value: plan.max_total_bytes.toLocaleString(),
    },
    { label: "发布状态", value: "生产发布已禁用" },
  ];
}

function stageIndex(planKind: string): number {
  const kind = planKind.toUpperCase();
  if (kind.includes("DELIVERY")) return 6;
  if (kind.includes("GRADING")) return 5;
  if (kind.includes("CRITERIA") || kind.includes("RUBRIC")) return 4;
  if (kind.includes("ATTACHMENT")) return 3;
  if (kind.includes("TASK")) return 2;
  if (kind.includes("TRACE")) return 1;
  return 0;
}

function humanize(value: string): string {
  return value
    .replaceAll("_", " ")
    .toLocaleLowerCase()
    .replace(/\b\w/g, (letter) => letter.toLocaleUpperCase());
}

function shortIdentity(value: string, max = 28): string {
  if (value.length <= max) return value;
  const keep = Math.max(6, Math.floor((max - 3) / 2));
  return `${value.slice(0, keep)}...${value.slice(-keep)}`;
}

function message(reason: unknown): string {
  if (reason instanceof ApiError) {
    return `${reason.code}: ${reason.message}`;
  }
  if (reason instanceof Error) return reason.message;
  return "控制台发生未知错误。";
}
