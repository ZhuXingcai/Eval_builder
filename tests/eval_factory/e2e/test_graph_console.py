from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fixture_builder import (
    USER,
    BrowserFixture,
    build_browser_fixture,
)
from harness import (
    BrowserDiagnostics,
    BrowserE2EPreflightError,
    BrowserE2EServerError,
    BrowserSession,
    ManagedServer,
    free_port,
    require_chromium,
)
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
FORBIDDEN = (
    "private-reference",
    "grader-rule",
    "hidden-condition",
    "final-answer",
    "model-output-body",
    "credential",
    "raw-trace",
)


@pytest.fixture(scope="module")
def fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> BrowserFixture:
    return build_browser_fixture(tmp_path_factory.mktemp("graph-console-fixture"))


@pytest.fixture(scope="module")
def session(fixture: BrowserFixture) -> Iterator[BrowserSession]:
    api_port = free_port()
    web_port = free_port()
    logs = ROOT / "runs" / "browser-e2e" / "logs"
    api = ManagedServer(
        command=(
            "uv",
            "run",
            "evalfactory",
            "agent",
            "serve",
            "--factory-store",
            str(fixture.store_path),
            "--registry",
            str(fixture.registry_path),
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
        environment={"VITE_API_PROXY_TARGET": (f"http://127.0.0.1:{api_port}")},
    )
    playwright = sync_playwright().start()
    require_chromium(playwright)
    api.start()
    web.start()
    succeeded = False
    try:
        api.wait_ready("/api/plan-reviews/contract")
        web.wait_ready("/")
        browser = playwright.chromium.launch(headless=True)
        value = BrowserSession(
            browser=browser,
            api=api,
            web=web,
            web_url=f"http://127.0.0.1:{web_port}",
            api_url=f"http://127.0.0.1:{api_port}",
        )
        yield value
        value.close()
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
) -> Iterator[Page]:
    context = session.browser.new_context(accept_downloads=True)
    page = context.new_page()
    diagnostics = BrowserDiagnostics(ROOT / "runs" / "browser-e2e" / "artifacts" / name)
    if diagnostics.root.exists():
        shutil.rmtree(diagnostics.root)
    diagnostics.attach(page)
    try:
        page.goto(session.web_url)
        page.wait_for_load_state("networkidle")
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


def _select(
    page: Page,
    fixture: BrowserFixture,
    key: str,
) -> None:
    review_id = fixture.reviews[key]
    page.locator(f'[data-review-id="{review_id}"]').click()
    page.locator(
        f'[data-review-id="{review_id}"][aria-current="true"]',
    ).wait_for()
    page.locator("[data-authority-hash]").wait_for()
    page.get_by_role(
        "tab",
        name="计划 JSON",
    ).click()
    page.locator(
        f'textarea[data-editor-review-id="{review_id}"]',
    ).wait_for()


def _open_activity(page: Page) -> None:
    page.get_by_role(
        "button",
        name="审核动态",
        exact=False,
    ).click()
    page.get_by_role(
        "dialog",
        name="审核动态",
    ).wait_for()


def _assert_safe(page: Page) -> None:
    content = page.content().casefold()
    for marker in FORBIDDEN:
        assert marker not in content


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    (
        ("attachment_generation", "works", None),
        ("criteria_rubric", "max_model_tokens", 8000),
        (
            "grading_design",
            "minimum_confidence_basis_points",
            8500,
        ),
    ),
)
def test_domain_edit_and_resume(
    session: BrowserSession,
    fixture: BrowserFixture,
    kind: str,
    field: str,
    value: object,
) -> None:
    key = f"{kind}-edit"
    with scenario(session, key) as page:
        _select(page, fixture, key)
        editor = page.get_by_label("可编辑计划 JSON")
        projection = json.loads(editor.input_value())
        if field == "works":
            projection["works"][0]["max_attempts"] = 1
        else:
            projection[field] = value
        editor.fill(json.dumps(projection))
        page.get_by_role(
            "button",
            name="提交修改",
        ).click()
        page.get_by_text("后继计划已编译并记录").wait_for()
        page.get_by_role(
            "button",
            name="恢复图执行",
        ).click()
        page.get_by_text("图恢复权限已提交").wait_for()
        assert "RESUMED" in page.locator(".state-badge").inner_text()
        _assert_safe(page)


@pytest.mark.parametrize(
    "kind",
    (
        "attachment_generation",
        "criteria_rubric",
        "grading_design",
    ),
)
def test_domain_approve_and_resume(
    session: BrowserSession,
    fixture: BrowserFixture,
    kind: str,
) -> None:
    key = f"{kind}-approve"
    with scenario(session, key) as page:
        _select(page, fixture, key)
        page.get_by_role("button", name="批准").click()
        page.get_by_text("计划已批准").wait_for()
        page.get_by_role(
            "button",
            name="恢复图执行",
        ).click()
        page.get_by_text("图恢复权限已提交").wait_for()
        _assert_safe(page)


@pytest.mark.parametrize(
    ("kind", "action", "notice"),
    (
        (
            "attachment_generation",
            "拒绝",
            "计划已拒绝",
        ),
        ("criteria_rubric", "拒绝", "计划已拒绝"),
        ("grading_design", "拒绝", "计划已拒绝"),
        (
            "attachment_generation",
            "暂缓",
            "计划已暂缓",
        ),
        ("criteria_rubric", "暂缓", "计划已暂缓"),
        ("grading_design", "暂缓", "计划已暂缓"),
        (
            "attachment_generation",
            "补充材料",
            "已请求补充规划材料",
        ),
        (
            "criteria_rubric",
            "补充材料",
            "已请求补充规划材料",
        ),
        (
            "grading_design",
            "补充材料",
            "已请求补充规划材料",
        ),
    ),
)
def test_domain_terminal_actions(
    session: BrowserSession,
    fixture: BrowserFixture,
    kind: str,
    action: str,
    notice: str,
) -> None:
    suffix = {
        "拒绝": "reject",
        "暂缓": "defer",
        "补充材料": "request-more",
    }[action]
    key = f"{kind}-{suffix}"
    with scenario(session, key) as page:
        _select(page, fixture, key)
        _open_activity(page)
        page.get_by_role("button", name=action).click()
        page.get_by_text(notice).wait_for()
        _assert_safe(page)


def test_stale_two_context_conflict_has_one_authority(
    session: BrowserSession,
    fixture: BrowserFixture,
) -> None:
    key = "grading_design-conflict"
    first_context = session.browser.new_context()
    second_context = session.browser.new_context()
    first = first_context.new_page()
    second = second_context.new_page()
    try:
        for page in (first, second):
            page.goto(session.web_url)
            page.wait_for_load_state("networkidle")
            _select(page, fixture, key)
        first.get_by_role("button", name="批准").click()
        first.get_by_text("计划已批准").wait_for()
        second.get_by_role("button", name="批准").click()
        second.get_by_role("alert").get_by_text(
            "PLAN_REVIEW_CONFLICT",
            exact=False,
        ).wait_for()
        response = first_context.request.get(
            f"{session.api_url}/api/plan-reviews/show",
            params={
                "review_id": fixture.reviews[key],
            },
        )
        assert response.ok
        assert response.json()["result"]["state"] == "APPROVED"
    finally:
        first_context.close()
        second_context.close()


def test_reload_and_api_restart_preserve_current_review(
    session: BrowserSession,
    fixture: BrowserFixture,
) -> None:
    key = "attachment_generation-reload"
    with scenario(session, key) as page:
        _select(page, fixture, key)
        hash_before = page.locator("[data-authority-hash]").get_attribute(
            "data-authority-hash",
        )
        session.restart_api()
        page.reload()
        page.wait_for_load_state("networkidle")
        _select(page, fixture, key)
        assert (
            page.locator("[data-authority-hash]").get_attribute(
                "data-authority-hash",
            )
            == hash_before
        )


def test_safe_export_and_negative_authorization(
    session: BrowserSession,
    fixture: BrowserFixture,
) -> None:
    key = "grading_design-export"
    with scenario(session, key) as page:
        _select(page, fixture, key)
        _open_activity(page)
        with page.expect_download() as pending:
            page.get_by_role(
                "button",
                name="导出 JSON",
            ).click()
        download = pending.value
        payload = json.loads(Path(download.path()).read_text(encoding="utf-8"))
        assert payload["plan"]["object_sha256"] == payload["request"]["plan_ref"]["object_sha256"]
        serialized = json.dumps(payload).casefold()
        for marker in FORBIDDEN:
            assert marker not in serialized

        mismatch = page.request.post(
            f"{session.api_url}/api/plan-reviews/decision",
            headers={
                "X-Eval-Factory-Principal": USER,
            },
            data={
                "review_id": fixture.reviews["criteria_rubric-invalid"],
                "submission": {
                    "expected_plan_version": 1,
                    "decision": "APPROVE",
                    "decided_by": "user://other",
                    "reason_code": "BROWSER_NEGATIVE",
                    "idempotency_key": ("browser-principal-mismatch"),
                },
            },
        )
        assert mismatch.status == 403
        assert mismatch.json()["error_code"] == "PRINCIPAL_MISMATCH"
        private = page.request.get(f"{session.api_url}/api/private-cas")
        assert private.status == 404


def test_invalid_edit_is_visible_and_closed(
    session: BrowserSession,
    fixture: BrowserFixture,
) -> None:
    key = "criteria_rubric-invalid"
    with scenario(session, key) as page:
        _select(page, fixture, key)
        page.get_by_label("可编辑计划 JSON").fill("{}")
        page.get_by_role(
            "button",
            name="提交修改",
        ).click()
        page.get_by_role("alert").get_by_text(
            "INVALID_COMMAND",
            exact=False,
        ).wait_for()
        error = page.get_by_role("alert").inner_text()
        assert "Traceback" not in error
        assert "sqlite" not in error.casefold()


def test_missing_chromium_preflight_is_typed() -> None:
    class MissingChromium:
        executable_path = "/definitely/missing/chromium"

    class MissingPlaywright:
        chromium = MissingChromium()

    with pytest.raises(
        BrowserE2EPreflightError,
        match="CHROMIUM_NOT_INSTALLED",
    ):
        require_chromium(MissingPlaywright())  # type: ignore[arg-type]


def test_server_timeout_is_typed_and_cleanup_is_exact(
    tmp_path: Path,
) -> None:
    server = ManagedServer(
        command=(
            "uv",
            "run",
            "python",
            "-c",
            "import time; time.sleep(60)",
        ),
        port=free_port(),
        log_path=tmp_path / "sleeping-server.log",
        cwd=ROOT,
    )
    server.start()
    try:
        with pytest.raises(
            BrowserE2EServerError,
            match="SERVER_READINESS_TIMEOUT",
        ):
            server.wait_ready(
                "/",
                timeout_seconds=0.2,
            )
    finally:
        server.stop()
    assert server.process is None


def test_network_diagnostics_are_bounded_and_query_free(
    tmp_path: Path,
) -> None:
    diagnostics = BrowserDiagnostics(tmp_path)
    for index in range(600):
        diagnostics._append_network(
            "GET",
            (f"http://127.0.0.1/api/plan-reviews?secret={index}"),
            200,
        )

    assert len(diagnostics.network) == 500
    assert all(value["path"] == "/api/plan-reviews" for value in diagnostics.network)
