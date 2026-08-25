from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from playwright.sync_api import Page, sync_playwright
from tests.eval_factory.e2e.harness import (
    BrowserDiagnostics,
    BrowserSession,
    ManagedServer,
    free_port,
    require_chromium,
)
from tests.eval_factory.integration.test_agent_shell_graph_flow import (
    RAW_ROOT,
    _real_trace_root,
)
from tests.integration.test_evalfactory_agent_serve_cli import (
    _host_config,
    _paths,
)

ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = (
    "private-reference",
    "grader-rule",
    "hidden-condition",
    "final-answer",
    "model-output-body",
    "credential",
    "raw-trace",
    "physical_path",
)

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


@pytest.fixture(scope="module")
def shell_session(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[BrowserSession, Path, Path]]:
    root = tmp_path_factory.mktemp("agent-shell-browser")
    paths = _paths(root)
    config_path = root / "shell-config.json"
    config_path.write_text(
        _host_config().model_dump_json(indent=2),
        encoding="utf-8",
    )
    source_root = root / "source-fixture"
    source_root.mkdir()
    manifest_path, _manifest_sha256 = _real_trace_root(
        source_root,
        instance_ids=("LH_077",),
    )
    trace_path = next(manifest_path.parent.glob("*.jsonl"))
    api_port = free_port()
    web_port = free_port()
    logs = ROOT / "runs" / "agent-shell-browser-e2e" / "logs"
    api = ManagedServer(
        command=(
            "uv",
            "run",
            "evalfactory",
            "agent",
            "serve",
            "--factory-store",
            str(paths["factory_store"]),
            "--shell-config",
            str(config_path),
            "--private-store",
            str(paths["private_store"]),
            "--gateway-store",
            str(paths["gateway_store"]),
            "--harness-store",
            str(paths["harness_store"]),
            "--source-store",
            str(paths["source_store"]),
            "--team-store",
            str(paths["team_store"]),
            "--capability-request-store",
            str(paths["capability_request_store"]),
            "--graph-journal",
            str(paths["graph_journal"]),
            "--graph-checkpoint",
            str(paths["graph_checkpoint"]),
            "--core-workspace",
            str(paths["core_workspace"]),
            "--attachment-workspace",
            str(paths["attachment_workspace"]),
            "--job-store",
            str(paths["job_store"]),
            "--specialist-workspace",
            str(paths["specialist_workspace"]),
            "--candidate-output",
            str(paths["candidate_output"]),
            "--user",
            "user://agent-shell",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
        ),
        port=api_port,
        log_path=logs / "api.log",
        cwd=ROOT,
    )
    web = ManagedServer(
        command=(
            "npm",
            "run",
            "dev",
            "--prefix",
            "web/eval_factory_console",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(web_port),
        ),
        port=web_port,
        log_path=logs / "vite.log",
        cwd=ROOT,
        environment={
            "VITE_API_PROXY_TARGET": f"http://127.0.0.1:{api_port}",
        },
    )
    playwright = sync_playwright().start()
    require_chromium(playwright)
    api.start()
    web.start()
    succeeded = False
    try:
        api.wait_ready("/api/harness/contract")
        web.wait_ready("/")
        browser = playwright.chromium.launch(headless=True)
        session = BrowserSession(
            browser=browser,
            api=api,
            web=web,
            web_url=f"http://127.0.0.1:{web_port}",
            api_url=f"http://127.0.0.1:{api_port}",
        )
        yield session, manifest_path, trace_path
        session.close()
        succeeded = True
    finally:
        web.stop()
        api.stop()
        playwright.stop()
        if succeeded:
            shutil.rmtree(logs, ignore_errors=True)


@contextmanager
def scenario(
    session: BrowserSession,
    name: str,
    *,
    width: int = 1440,
    height: int = 960,
) -> Iterator[Page]:
    context = session.browser.new_context(
        viewport={"width": width, "height": height},
    )
    page = context.new_page()
    diagnostics = BrowserDiagnostics(
        ROOT / "runs" / "agent-shell-browser-e2e" / "artifacts" / name,
    )
    if diagnostics.root.exists():
        shutil.rmtree(diagnostics.root)
    diagnostics.attach(page)
    try:
        page.goto(session.web_url, wait_until="domcontentloaded")
        page.locator(".agent-shell").wait_for()
        yield page
    except Exception:
        diagnostics.write_failure(
            page,
            api_log=session.api.log_path,
            web_log=session.web.log_path,
        )
        raise
    finally:
        context.close()


def test_conversation_source_graph_review_and_team(
    shell_session: tuple[BrowserSession, Path, Path],
) -> None:
    session, manifest_path, trace_path = shell_session
    with scenario(session, "conversation-flow") as page:
        _create_new_session(page)
        page.get_by_label("输入评测需求").wait_for()
        page.get_by_role("button", name="添加上下文").click()
        page.get_by_role("button", name="文件和文件夹").click()
        page.get_by_label("选择来源文件", exact=True).set_input_files(
            [manifest_path, trace_path],
        )
        page.get_by_role("button", name="确认接入").click()
        page.get_by_text("来源已接入").wait_for(timeout=15_000)
        composer = page.get_by_label("输入评测需求")
        composer.focus()
        focus_style = composer.evaluate(
            """
            (element) => {
              const style = getComputedStyle(element);
              return {
                outline: style.outlineStyle,
                shadow: style.boxShadow,
              };
            }
            """
        )
        assert focus_style == {"outline": "none", "shadow": "none"}
        textarea_box = composer.bounding_box()
        add_box = page.get_by_role(
            "button",
            name="添加上下文",
        ).bounding_box()
        send_box = page.get_by_role(
            "button",
            name="发送需求",
        ).bounding_box()
        assert textarea_box is not None
        assert add_box is not None
        assert send_box is not None
        assert add_box["y"] > textarea_box["y"]
        assert send_box["y"] > textarea_box["y"]
        composer.fill("构建一套可审计的通用 Agent 评测数据。")
        page.get_by_role("button", name="发送需求").click()
        page.get_by_text("等待审核").wait_for(timeout=30_000)
        quote = page.get_by_role(
            "button",
            name="引用 Agent 输出到输入框",
        ).first
        quote.click()
        page.get_by_text("Requirement is ready.").last.wait_for()
        page.get_by_role(
            "button",
            name="移除 Agent 输出引用",
        ).click()
        page.get_by_role("button", name="打开审核").click()
        page.wait_for_url("**/workspace/plan_review")
        page.get_by_role("heading", name="权限概览").wait_for()
        page.locator(
            ".embedded-review-shell [data-authority-hash]",
        ).wait_for()
        page.get_by_role("button", name="批准").evaluate(
            "(button) => button.click()",
        )
        page.get_by_text("计划已批准").wait_for()
        page.get_by_role("button", name="恢复图执行").click()
        page.get_by_text("图恢复权限已提交").wait_for()
        page.get_by_text("执行状态已同步").wait_for(timeout=30_000)
        page.locator(".header-actions").get_by_role(
            "button",
            name="团队",
            exact=True,
        ).click()
        page.get_by_role("dialog", name="团队面板").wait_for()
        assert page.locator(".inspector-members button").count() >= 1
        _assert_safe(page)


def test_deep_link_reload_and_sse_reconnect(
    shell_session: tuple[BrowserSession, Path, Path],
) -> None:
    session, _manifest_path, _trace_path = shell_session
    with scenario(session, "deep-link-reconnect") as page:
        _wait_for_session(page)
        page.locator(".workspace-navigation button").filter(
            has=page.get_by_text("评测工作台", exact=True),
        ).click()
        page.wait_for_url("**/workspace/trace")
        page.get_by_role("heading", name="轨迹来源").wait_for()
        route = page.url

        page.reload(wait_until="domcontentloaded")
        _wait_for_session(page)
        page.get_by_role("heading", name="轨迹来源").wait_for()
        assert page.url == route

        page.locator(".stream-state.online").wait_for(timeout=20_000)
        session.api.stop()
        try:
            page.locator(
                ".stream-state.offline, .stream-state.reconnecting",
            ).wait_for(timeout=25_000)
        finally:
            session.api.start()
            session.api.wait_ready("/api/harness/contract")
        page.locator(".stream-state.online").wait_for(timeout=30_000)
        assert page.url == route
        delivery = page.get_by_role(
            "navigation",
            name="评测工作台视图",
        ).get_by_role("button", name="交付", exact=True)
        assert delivery.get_attribute("data-workspace-status") == "BLOCKED"
        assert delivery.is_disabled()
        assert delivery.get_attribute("title") == "EXECUTION_BLOCKED"
        page.goto(
            route.rsplit("/workspace/", maxsplit=1)[0] + "/workspace/delivery",
            wait_until="domcontentloaded",
        )
        _wait_for_session(page)
        page.get_by_text("交付尚未生成").wait_for()
        _assert_safe(page)


def test_mobile_drawers_manage_focus_escape_and_reduced_motion(
    shell_session: tuple[BrowserSession, Path, Path],
) -> None:
    session, _manifest_path, _trace_path = shell_session
    with scenario(
        session,
        "mobile-accessibility",
        width=375,
        height=812,
    ) as page:
        _wait_for_session(page)
        page.emulate_media(reduced_motion="reduce")
        transition_ms = page.locator(".shell-icon-button").first.evaluate(
            """
            (element) => {
              const value = getComputedStyle(element).transitionDuration;
              return value.endsWith("ms")
                ? Number.parseFloat(value)
                : Number.parseFloat(value) * 1000;
            }
            """
        )
        assert transition_ms <= 0.01

        menu = page.get_by_role("button", name="打开会话导航")
        menu.click()
        navigation = page.get_by_role("dialog", name="会话导航")
        navigation.wait_for()
        navigation.locator(
            'button[aria-label="关闭会话导航"]:focus',
        ).wait_for()
        page.keyboard.press("Tab")
        assert navigation.locator(":focus").count() == 1
        page.keyboard.press("Escape")
        navigation.wait_for(state="hidden")
        page.locator(
            'button[aria-label="打开会话导航"]:focus',
        ).wait_for()

        activity = page.get_by_role("button", name="活动", exact=True)
        activity.click()
        inspector = page.get_by_role("dialog", name="活动面板")
        inspector.wait_for()
        inspector.locator(
            'button[aria-label="关闭检查器"]:focus',
        ).wait_for()
        page.keyboard.press("Shift+Tab")
        assert inspector.locator(":focus").count() == 1
        page.keyboard.press("Escape")
        inspector.wait_for(state="hidden")
        page.locator('button[aria-label="活动"]:focus').wait_for()
        _assert_safe(page)


def test_session_removal_confirms_and_returns_focus(
    shell_session: tuple[BrowserSession, Path, Path],
) -> None:
    session, _manifest_path, _trace_path = shell_session
    with scenario(session, "session-removal") as page:
        _create_new_session(page)
        delete = page.locator(
            ".session-row-shell.active .session-delete-button",
        )
        delete.click()
        dialog = page.locator("#close-session-dialog")
        dialog.wait_for()
        page.keyboard.press("Escape")
        dialog.wait_for(state="hidden")
        page.locator(
            ".session-row-shell.active .session-delete-button:focus",
        ).wait_for()

        delete.click()
        page.get_by_role(
            "button",
            name="删除会话",
            exact=True,
        ).click()
        dialog.wait_for(state="hidden")
        page.get_by_text("会话已从列表移除").wait_for()
        statuses = page.evaluate(
            """
            async () => {
              const response = await fetch("/api/harness/sessions?limit=100");
              const body = await response.json();
              return body.sessions.map((item) => item.status);
            }
            """
        )
        assert "CLOSED" not in statuses
        _assert_safe(page)


def test_model_picker_combines_effort_and_persists_custom_models(
    shell_session: tuple[BrowserSession, Path, Path],
) -> None:
    session, _manifest_path, _trace_path = shell_session
    with scenario(session, "model-picker") as page:
        _create_new_session(page)
        trigger = page.get_by_role(
            "button",
            name="选择模型和思考深度",
        )
        trigger.click()
        picker = page.get_by_role(
            "dialog",
            name="模型和思考深度",
        )
        picker.wait_for()
        picker.locator(".model-option").filter(has_text="Claude").click()
        effort = picker.get_by_label("Claude 思考深度")
        effort.fill("4")
        assert effort.get_attribute("aria-valuetext") == "Max"
        page.screenshot(
            path="/tmp/eval-agent-model-picker-1440.png",
            full_page=True,
        )

        picker.get_by_role("button", name="自定义模型").click()
        form = page.get_by_role("dialog", name="添加自定义模型")
        form.get_by_label("显示名称").fill("团队推理模型")
        form.get_by_label("服务商").fill("OpenRouter")
        form.get_by_label("模型 ID").fill("vendor/reasoner-v1")
        form.get_by_label("Base URL").fill(
            "https://gateway.example.com/v1",
        )
        form.get_by_label("凭证环境变量").fill("TEAM_MODEL_API_KEY")
        form.get_by_label("最高思考深度").select_option("XHIGH")
        page.screenshot(
            path="/tmp/eval-agent-custom-model-1440.png",
            full_page=True,
        )
        form.get_by_role("button", name="添加模型").click()
        picker.wait_for()
        assert "团队推理模型" in trigger.text_content()
        page.get_by_role("button", name="关闭模型选择").click()

        page.reload(wait_until="domcontentloaded")
        page.locator(".agent-heading strong").wait_for()
        trigger = page.get_by_role(
            "button",
            name="选择模型和思考深度",
        )
        trigger.filter(has_text="团队推理模型").wait_for()
        trigger.click()
        page.get_by_role(
            "button",
            name="删除自定义模型 团队推理模型",
        ).click()
        assert "自动模型" in trigger.text_content()
        _assert_safe(page)

    with scenario(
        session,
        "model-picker-mobile",
        width=375,
        height=812,
    ) as page:
        page.locator(".stream-state.online").wait_for(timeout=20_000)
        page.get_by_role("button", name="打开会话导航").click()
        _create_new_session(page)
        page.get_by_role(
            "dialog",
            name="会话导航",
        ).wait_for(state="hidden")
        page.wait_for_timeout(250)
        page.get_by_role(
            "button",
            name="选择模型和思考深度",
        ).click()
        picker = page.get_by_role(
            "dialog",
            name="模型和思考深度",
        )
        picker.wait_for()
        bounds = picker.bounding_box()
        assert bounds is not None
        assert bounds["x"] >= 0
        assert bounds["x"] + bounds["width"] <= 375
        assert bounds["y"] >= 0
        assert bounds["y"] + bounds["height"] <= 812
        page.screenshot(
            path="/tmp/eval-agent-model-picker-375.png",
            full_page=True,
        )
        _assert_safe(page)


@pytest.mark.parametrize(
    ("width", "height"),
    (
        (1440, 960),
        (1024, 768),
        (768, 900),
        (375, 812),
    ),
)
def test_agent_shell_responsive_layout(
    shell_session: tuple[BrowserSession, Path, Path],
    width: int,
    height: int,
) -> None:
    session, _manifest_path, _trace_path = shell_session
    with scenario(
        session,
        f"responsive-{width}",
        width=width,
        height=height,
    ) as page:
        if width <= 900:
            page.locator(".stream-state.online").wait_for(timeout=20_000)
            page.get_by_role("button", name="打开会话导航").click()
            page.get_by_role("dialog", name="会话导航").wait_for()
        _create_new_session(page)
        metrics = page.evaluate(
            """
            () => ({
              viewport: document.documentElement.clientWidth,
              scroll: document.documentElement.scrollWidth,
              bodyScroll: document.body.scrollWidth,
            })
            """
        )
        assert metrics["scroll"] <= metrics["viewport"]
        assert metrics["bodyScroll"] <= metrics["viewport"]
        page.screenshot(
            path=f"/tmp/eval-agent-shell-{width}.png",
            full_page=True,
        )
        if width <= 900:
            page.get_by_role("button", name="打开会话导航").click()
            page.get_by_role("dialog", name="会话导航").wait_for()
        _assert_safe(page)


def _wait_for_session(page: Page) -> None:
    page.locator(".shell-status-bar code").filter(
        has_not_text="no session",
    ).wait_for(state="attached", timeout=15_000)
    page.locator(".agent-heading strong").filter(
        has_not_text="新的评测任务",
    ).wait_for()


def _create_new_session(page: Page) -> None:
    previous_url = page.url
    page.get_by_role("button", name="新建会话").click()
    page.wait_for_function(
        r"""
        (previousUrl) =>
          window.location.href !== previousUrl &&
          /\/sessions\/[^/]+$/.test(
            window.location.pathname
          )
        """,
        arg=previous_url,
    )
    page.locator(".agent-main.composer-centered").wait_for()


def _assert_safe(page: Page) -> None:
    content = page.content().casefold()
    for marker in FORBIDDEN:
        assert marker not in content
