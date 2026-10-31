"""Startup validation for the dashboard webhook configuration: actionable warnings, never secrets."""

from tests.fakes import make_settings


def test_no_warnings_when_outbound_is_disabled_and_nothing_is_set():
    settings = make_settings(enable_outbound_calls=False)
    assert settings.webhook_configuration_warnings() == []


def test_warns_when_outbound_is_enabled_but_the_secret_is_missing():
    settings = make_settings(enable_outbound_calls=True)
    warnings = settings.webhook_configuration_warnings()
    assert len(warnings) == 1
    assert "DASHBOARD_WEBHOOK_SECRET" in warnings[0] and "campaign dispatch" in warnings[0].lower()


def test_no_outbound_warning_once_the_secret_is_set():
    settings = make_settings(enable_outbound_calls=True, dashboard_webhook_secret="s")
    warnings = settings.webhook_configuration_warnings()
    assert all("every campaign dispatch will be rejected" not in w for w in warnings)


def test_warns_when_the_inbound_report_url_is_set_without_a_secret():
    settings = make_settings(dashboard_webhook_url="https://dashboard.test/api/webhooks/voice-engine")
    warnings = settings.webhook_configuration_warnings()
    assert len(warnings) == 1
    assert "DASHBOARD_WEBHOOK_URL" in warnings[0] and "DASHBOARD_WEBHOOK_SECRET" in warnings[0]


def test_warns_when_the_secret_is_set_without_an_inbound_report_url():
    settings = make_settings(dashboard_webhook_secret="s")
    warnings = settings.webhook_configuration_warnings()
    assert len(warnings) == 1
    assert "DASHBOARD_WEBHOOK_URL" in warnings[0]


def test_no_warnings_once_both_halves_are_configured():
    settings = make_settings(
        enable_outbound_calls=True,
        dashboard_webhook_secret="s",
        dashboard_webhook_url="https://dashboard.test/api/webhooks/voice-engine",
    )
    assert settings.webhook_configuration_warnings() == []


def test_never_includes_the_secret_value_itself():
    settings = make_settings(enable_outbound_calls=True, dashboard_webhook_secret="super-secret-value")
    warnings = settings.webhook_configuration_warnings()
    assert all("super-secret-value" not in w for w in warnings)
