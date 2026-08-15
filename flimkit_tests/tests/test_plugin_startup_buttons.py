import pytest

from flimkit import plugins
from flimkit.plugins import registry


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear()
    yield
    registry.clear()


def test_startup_runs_with_the_app():
    seen = []

    @plugins.startup('demo_startup')
    def start(app):
        seen.append(app)

    app = object()
    failures = plugins.run_startups(app)

    assert failures == []
    assert seen == [app]


def test_startup_failure_is_reported_not_raised():
    @plugins.startup('broken_startup')
    def start(app):
        raise RuntimeError('no port free')

    failures = plugins.run_startups(object())

    assert len(failures) == 1
    entry, exc = failures[0]
    assert entry.id == 'broken_startup'
    assert isinstance(exc, RuntimeError)


def test_startups_run_in_order():
    order = []

    @plugins.startup('second', order=200)
    def second(app):
        order.append('second')

    @plugins.startup('first', order=100)
    def first(app):
        order.append('first')

    plugins.run_startups(object())

    assert order == ['first', 'second']


def test_duplicate_startup_id_is_refused():
    @plugins.startup('same')
    def one(app):
        pass

    with pytest.raises(plugins.PluginError, match='already registered'):
        @plugins.startup('same')
        def two(app):
            pass


def test_panel_button_registers_and_lists():
    @plugins.panel_button('send_demo', 'Send to Demo', panel='roi')
    def send(app):
        return 'sent'

    found = plugins.panel_buttons('roi')

    assert [b.id for b in found] == ['send_demo']
    assert found[0].label == 'Send to Demo'
    assert found[0].callback(object()) == 'sent'


def test_panel_button_rejects_unknown_panel():
    with pytest.raises(plugins.PluginError, match='unknown panel'):
        @plugins.panel_button('bad', 'Bad', panel='nowhere')
        def bad(app):
            pass


def test_panel_buttons_sort_by_order():
    @plugins.panel_button('b', 'B', order=200)
    def b(app):
        pass

    @plugins.panel_button('a', 'A', order=100)
    def a(app):
        pass

    assert [x.id for x in plugins.panel_buttons('roi')] == ['a', 'b']


def test_clear_removes_startups_and_buttons():
    @plugins.startup('s')
    def s(app):
        pass

    @plugins.panel_button('p', 'P')
    def p(app):
        pass

    registry.clear()

    assert plugins.startups() == []
    assert plugins.panel_buttons() == []
