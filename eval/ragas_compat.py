import sys
import types


def ensure_ragas_import_compat() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - compatibility shim only
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim
