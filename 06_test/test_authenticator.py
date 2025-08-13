import pytest
from authenticator import Authenticator

@pytest.fixture
def authenticator():
    authenticator = Authenticator()
    yield authenticator

# ユーザー登録のテスト
@pytest.mark.parametrize("username, password", [
    ("user1", "pass1"),
    ("user2", "pass2"),
    ("user3", "pass3"),
])
def test_register(authenticator, username, password):
    authenticator.register(username, password)
    assert username in authenticator.users
    assert authenticator.users[username] == password

# 既存ユーザー登録のテスト
@pytest.mark.parametrize("username, password", [
    ("user1", "pass1"),
    ("user2", "pass2"),
])
def test_register_existing_user(authenticator, username, password):
    authenticator.register(username, password)
    with pytest.raises(ValueError, match="エラー: ユーザーは既に存在します。"):
        authenticator.register(username, password)

# ログインのテスト
@pytest.mark.parametrize("username, password, expected", [
    ("user1", "pass1", "ログイン成功"),
    ("user2", "pass2", "ログイン成功"),
    ("user3", "pass3", "ログイン成功"),
])
def test_login(authenticator, username, password, expected):
    authenticator.register(username, password)
    result = authenticator.login(username, password)
    assert result == expected

# ログイン失敗のテスト
@pytest.mark.parametrize("username, password", [
    ("user1", "wrongpass"),
    ("user2", "wrongpass"),
    ("nonexistent", "pass"),
])
def test_login_failure(authenticator, username, password):
    authenticator.register("user1", "hoge")
    authenticator.register("user2", "fuga")
    with pytest.raises(ValueError, match="エラー: ユーザー名またはパスワードが正しくありません。"):
        authenticator.login(username, password)