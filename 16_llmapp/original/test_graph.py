import pytest
from langchain_openai import OpenAIEmbeddings
from original.graph import (
    get_bot_response,
    get_messages_list,
    memory,
    build_graph,
    define_tools,
    create_index,
    robot_fortune_telling,
)

# モック用のテストデータ
USER_MESSAGE_1 = "1たす2は？"
USER_MESSAGE_2 = "東京駅のイベントの検索結果を教えて"
USER_MESSAGE_3 = "有給休暇の日数は？"
USER_MESSAGE_FORTUNE = "占いをお願いします"
THREAD_ID = "test_thread"
 
@pytest.fixture
def setup_memory():
    """
    テスト用のメモリを初期化。
    """
    memory.storage.clear()
    return memory

@pytest.fixture
def setup_graph():
    """
    テスト用に新しいグラフを構築。
    """
    return build_graph("gpt-4o-mini", memory)

def test_define_tools():
    """
    define_tools関数が正しくツールを定義できるかをテスト。
    """
    tools = define_tools()
    assert len(tools) >= 3, "ツールが正しく定義される必要があります（retriever、tavily、robot_fortune_telling）。"
    assert any(tool.name == "retrieve_company_rules" for tool in tools), "retrieve_company_rulesツールが定義されるべきです。"
    assert any(tool.name == "robot_fortune_telling" for tool in tools), "robot_fortune_tellingツールが定義されるべきです。"

def test_create_index():
    """
    create_index関数がインデックスを正しく構築するかをテスト。
    """
    persist_directory = "./test_chroma_db"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        index = create_index(persist_directory, embedding_model)
        assert index is not None, "インデックスが作成されるべきです。"
    except Exception as e:
        pytest.fail(f"インデックス作成中にエラーが発生しました: {e}")

def test_get_bot_response_single_message(setup_memory):
    """
    ボットがシンプルなメッセージに応答できるかをテスト。
    """
    response = get_bot_response(USER_MESSAGE_1, setup_memory, THREAD_ID)
    assert isinstance(response, str), "応答は文字列である必要があります。"
    assert "3" in response, "1たす2の計算結果が正しく応答されるべきです。"

def test_get_bot_response_with_rag(setup_memory):
    """
    ボットがRetrieverベースの質問応答を正しく処理できるかをテスト。
    """
    response = get_bot_response(USER_MESSAGE_3, setup_memory, THREAD_ID)
    assert isinstance(response, str), "応答は文字列である必要があります。"
    assert "有給休暇" in response, "RAGベースの質問に関連する応答が含まれるべきです。"

def test_get_bot_response_multiple_messages(setup_memory):
    """
    複数のメッセージを処理してメモリに保存されるかをテスト。
    """
    get_bot_response(USER_MESSAGE_1, setup_memory, THREAD_ID)
    get_bot_response(USER_MESSAGE_2, setup_memory, THREAD_ID)
    messages = get_messages_list(setup_memory, THREAD_ID)
    assert len(messages) >= 2, "メモリに2つ以上のメッセージが保存されるべきです。"
    assert any("1たす2" in msg['text'] for msg in messages if msg['class'] == 'user-message'), "メモリに最初のユーザーメッセージが保存されるべきです。"
    assert any("東京駅" in msg['text'] for msg in messages if msg['class'] == 'user-message'), "メモリに2番目のユーザーメッセージが保存されるべきです。"

def test_memory_clear_on_new_session(setup_memory):
    """
    新しいセッションでメモリがクリアされるかをテスト。
    """
    get_bot_response(USER_MESSAGE_1, setup_memory, THREAD_ID)
    initial_messages = get_messages_list(setup_memory, THREAD_ID)
    assert len(initial_messages) > 0, "最初のメッセージがメモリに保存されていない可能性があります。"

    setup_memory.storage.clear()
    cleared_messages = setup_memory.get({"configurable": {"thread_id": THREAD_ID}})
    assert cleared_messages is None or 'channel_values' not in cleared_messages, "メモリがクリアされていません。"

def test_build_graph(setup_memory):
    """
    グラフが正しく構築され、応答を生成できるかをテスト。
    """
    graph = build_graph("gpt-4o-mini", setup_memory)
    response = graph.invoke(
        {"messages": [("user", USER_MESSAGE_1)]},
        {"configurable": {"thread_id": THREAD_ID}},
        stream_mode="values"
    )
    assert response["messages"][-1].content, "グラフが有効な応答を生成する必要があります。"

def test_get_messages_list(setup_memory):
    """
    メモリ内のメッセージリストが正しく取得されるかをテスト。
    """
    get_bot_response(USER_MESSAGE_1, setup_memory, THREAD_ID)
    messages = get_messages_list(setup_memory, THREAD_ID)
    assert len(messages) > 0, "応答後、メッセージリストは空であってはなりません。"
    assert any(isinstance(msg, dict) for msg in messages), "メッセージリストは辞書のリストである必要があります。"
    assert any(msg['class'] == 'user-message' for msg in messages), "メッセージリストにユーザーのメッセージが含まれている必要があります。"
    assert any(msg['class'] == 'bot-message' for msg in messages), "メッセージリストにボットの応答が含まれている必要があります。"

def test_robot_fortune_telling_tool():
    """
    ロボット占いツールが正しく動作するかをテスト。
    """
    result = robot_fortune_telling.invoke({})
    assert isinstance(result, str), "占い結果は文字列である必要があります。"
    
    # 結果に占い要素が含まれているかを確認
    fortune_results = ["大吉", "吉", "凶"] 
    has_fortune = any(fortune in result for fortune in fortune_results)
    assert has_fortune, "占い結果に大吉、吉、凶のいずれかが含まれるべきです。"

def test_robot_personality_in_responses(setup_memory):
    """
    ボットの応答にロボット調の特徴が含まれているかをテスト。
    """
    response = get_bot_response("こんにちは", setup_memory, THREAD_ID)
    assert isinstance(response, str), "応答は文字列である必要があります。"
    
    # ロボット調の特徴を確認
    robot_characteristics = ["デス", "シマス", "ピコーン", "ガガガ"]
    has_robot_talk = any(char in response for char in robot_characteristics)
    assert has_robot_talk, "ボットの応答にロボット調の特徴が含まれるべきです。"

# 実行用
if __name__ == "__main__":
    pytest.main()