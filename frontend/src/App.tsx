import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BookOpen,
  Braces,
  MessageSquarePlus,
  MoreHorizontal,
  PanelRight,
  PencilLine,
  Pin,
  Send,
  Trash2
} from "lucide-react";
import { api, streamChatMessage } from "./api";
import { AnswerMessage } from "./components/AnswerMessage";
import { MarkdownMessage } from "./components/MarkdownMessage";
import { ModeDropdown } from "./components/ModeDropdown";
import { SubjectDropdown } from "./components/SubjectDropdown";
import { useWorkbenchStore } from "./store";
import type {
  AgentExecutionStep,
  ChatSession,
  GraphPayload,
  MessageDetails,
  ModeId,
  RetrievedChunk,
  StreamEvent,
  SubjectId
} from "./types";

const SUBJECTS: Array<{ id: SubjectId; subjects: string[] }> = [
  { id: "auto", subjects: [] },
  { id: "C_program", subjects: ["C_program"] },
  { id: "operating_systems", subjects: ["operating_systems"] },
  { id: "cybersec_lab", subjects: ["cybersec_lab"] }
];

export function App() {
  const queryClient = useQueryClient();
  const abortRef = useRef<AbortController | null>(null);
  const [draft, setDraft] = useState("");
  const [codeOpen, setCodeOpen] = useState(false);
  const [codeQuestion, setCodeQuestion] = useState("");
  const [codeValue, setCodeValue] = useState("");
  const [openChatMenuId, setOpenChatMenuId] = useState<string | null>(null);
  const store = useWorkbenchStore();

  const chatsQuery = useQuery({ queryKey: ["chats"], queryFn: api.listChats });
  const activeChatQuery = useQuery({
    queryKey: ["chat", store.activeChatId],
    queryFn: () => api.getChat(store.activeChatId || ""),
    enabled: Boolean(store.activeChatId)
  });

  useEffect(() => {
    if (chatsQuery.data?.chats) {
      store.setChats(chatsQuery.data.chats);
      if (!store.activeChatId && chatsQuery.data.chats.length) {
        store.setActiveChat(chatsQuery.data.chats[0]);
      }
    }
  }, [chatsQuery.data]);

  useEffect(() => {
    if (activeChatQuery.data) store.setActiveChat(activeChatQuery.data);
  }, [activeChatQuery.data]);

  useEffect(() => {
    function onPointerDown(event: PointerEvent) {
      const target = event.target as Element | null;
      if (!target?.closest(".chat-menu-wrap")) setOpenChatMenuId(null);
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpenChatMenuId(null);
    }

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  const createChat = useMutation({
    mutationFn: () => api.createChat(store.preferredMode),
    onSuccess: (chat) => {
      store.setActiveChat(chat);
      queryClient.invalidateQueries({ queryKey: ["chats"] });
    }
  });

  const currentSubject = useMemo(
    () => SUBJECTS.find((item) => item.id === store.preferredSubject) || SUBJECTS[0],
    [store.preferredSubject]
  );

  async function ensureChat(): Promise<ChatSession> {
    if (store.activeChatId) {
      return activeChatQuery.data || (await api.getChat(store.activeChatId));
    }
    const chat = await api.createChat(store.preferredMode);
    store.setActiveChat(chat);
    queryClient.invalidateQueries({ queryKey: ["chats"] });
    return chat;
  }

  async function sendMessage(extraPayload: Record<string, unknown> = {}) {
    const text = String(extraPayload.message || draft).trim();
    if (!text || store.sending) return;
    const chat = await ensureChat();
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const effectiveSubject = effectiveSubjectForMessage(extraPayload, store.preferredSubject);
    store.appendUserMessage(text);
    store.startAssistantMessage(store.preferredMode, effectiveSubject);
    store.setSending(true);
    setDraft("");

    try {
      await streamChatMessage(
        chat.chat_id,
        {
          message: text,
          mode: store.preferredMode,
          subjects: currentSubject.subjects,
          ...extraPayload
        },
        abortRef.current.signal,
        handleStreamEvent
      );
      queryClient.invalidateQueries({ queryKey: ["chats"] });
      queryClient.invalidateQueries({ queryKey: ["chat", chat.chat_id] });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      store.appendAssistantDelta(`\n\n请求失败：${message}`);
      store.markCurrentAssistantError(message);
    } finally {
      store.setSending(false);
    }
  }

  function handleStreamEvent(event: StreamEvent) {
    if (event.event === "delta") {
      store.appendAssistantDelta(String(event.data.text || ""));
      return;
    }
    if (event.event === "meta") {
      store.updateCurrentMeta(event.data);
      return;
    }
    if (
      event.event === "workflow_node_start" ||
      event.event === "workflow_node_end" ||
      event.event === "workflow_node_error"
    ) {
      store.updateCurrentWorkflowStep(event.data as AgentExecutionStep);
      return;
    }
    if (event.event === "graph_update") {
      store.updateCurrentGraph(event.data as GraphPayload);
      return;
    }
    if (event.event === "chunks_update") {
      store.updateCurrentChunks((event.data.chunks || []) as RetrievedChunk[]);
      return;
    }
    if (event.event === "done") {
      const answer = typeof event.data.answer === "string" ? event.data.answer : "";
      const meta = typeof event.data.assistant_meta === "string" ? event.data.assistant_meta : "";
      const details = isMessageDetails(event.data.message_details) ? event.data.message_details : undefined;
      if (answer) store.finishAssistantMessage(answer, meta, details);
    }
  }

  async function renameChat(chat: ChatSession) {
    const nextTitle = window.prompt("重命名会话", chat.title);
    if (!nextTitle) return;
    await api.renameChat(chat.chat_id, nextTitle);
    queryClient.invalidateQueries({ queryKey: ["chats"] });
    queryClient.invalidateQueries({ queryKey: ["chat", chat.chat_id] });
  }

  async function deleteChat(chat: ChatSession) {
    if (!window.confirm(`删除「${chat.title}」？`)) return;
    await api.deleteChat(chat.chat_id);
    if (store.activeChatId === chat.chat_id) store.setActiveChat(null);
    queryClient.invalidateQueries({ queryKey: ["chats"] });
  }

  async function togglePin(chat: ChatSession) {
    await api.pinChat(chat.chat_id, !chat.pinned);
    queryClient.invalidateQueries({ queryKey: ["chats"] });
  }

  async function updateMode(mode: ModeId) {
    store.setPreferredMode(mode);
    if (store.activeChatId) await api.setMode(store.activeChatId, mode);
  }

  async function submitCodeAnalysis() {
    const code = codeValue.trim();
    if (!code) return;
    const message = `${codeQuestion || "请分析这段 C 代码"}\n\n\`\`\`c\n${code}\n\`\`\``;
    setCodeOpen(false);
    setCodeValue("");
    setCodeQuestion("");
    await sendMessage({
      message,
      code_analysis: true,
      subjects: ["C_program"]
    });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">GM</div>
          <div>
            <div className="brand-title">GraphMind Tutor</div>
            <div className="brand-subtitle">图思助教</div>
          </div>
        </div>
        <button className="primary-action" onClick={() => createChat.mutate()}>
          <MessageSquarePlus size={18} />
          新建聊天
        </button>
        <div className="chat-list">
          {store.chats.map((chat) => (
            <div
              key={chat.chat_id}
              className={`chat-row ${store.activeChatId === chat.chat_id ? "active" : ""}`}
              onClick={() => {
                setOpenChatMenuId(null);
                store.setActiveChat(chat);
              }}
            >
              <span className="min-w-0 flex-1 truncate text-left">{chat.title}</span>
              <div className="chat-menu-wrap" onClick={(event) => event.stopPropagation()}>
                <button
                  aria-expanded={openChatMenuId === chat.chat_id}
                  aria-label="会话操作"
                  className="chat-menu-trigger"
                  type="button"
                  onClick={() =>
                    setOpenChatMenuId((current) =>
                      current === chat.chat_id ? null : chat.chat_id
                    )
                  }
                >
                  <MoreHorizontal size={16} />
                </button>
                {openChatMenuId === chat.chat_id ? (
                  <div className="chat-menu">
                    <button
                      type="button"
                      onClick={() => {
                        setOpenChatMenuId(null);
                        togglePin(chat);
                      }}
                    >
                      <Pin size={14} />
                      {chat.pinned ? "取消固定" : "固定会话"}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setOpenChatMenuId(null);
                        renameChat(chat);
                      }}
                    >
                      <PencilLine size={14} />
                      重命名
                    </button>
                    <button
                      className="danger"
                      type="button"
                      onClick={() => {
                        setOpenChatMenuId(null);
                        deleteChat(chat);
                      }}
                    >
                      <Trash2 size={14} />
                      删除
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
        </div>
        <a className="legacy-link" href="/legacy">旧版页面</a>
      </aside>

      <main className="chat-surface">
        <header className="workspace-toolbar">
          <div className="workspace-title-block">
            <div>
              <div className="workspace-kicker">GraphMind Tutor</div>
              <h1 className="workspace-title">图思助教</h1>
            </div>
            <div className="toolbar-controls">
              <ModeDropdown value={store.preferredMode} onChange={updateMode} />
              <SubjectDropdown value={store.preferredSubject} onChange={store.setPreferredSubject} />
            </div>
          </div>
        </header>

        <section className="message-pane">
          {store.activeMessages.length ? (
            store.activeMessages.map((message, index) =>
              message.role === "assistant" ? (
                <AnswerMessage key={`${message.role}-${index}`} message={message} />
              ) : (
                <article key={`${message.role}-${index}`} className="message user">
                  <MarkdownMessage content={message.content} />
                </article>
              )
            )
          ) : (
            <div className="empty-state">
              <BookOpen size={38} />
              <div className="mt-4 text-xl font-semibold">选择课程问题开始</div>
              <p className="mt-2 max-w-md text-sm leading-6 text-stone-500">
                回答框内会保留本次检索链路、知识图谱命中和引用证据。
              </p>
            </div>
          )}
        </section>

        <footer className="composer">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
            placeholder="输入课程问题，按 Enter 发送"
          />
          <div className="composer-actions">
            <button
              className="ghost-action"
              disabled={store.preferredSubject !== "C_program" || store.sending}
              onClick={() => setCodeOpen(true)}
            >
              <Braces size={17} />
              代码分析
            </button>
            <button
              className="ghost-action"
              disabled={store.sending}
              onClick={() => sendMessage({ problem_tutoring: true })}
            >
              <PanelRight size={17} />
              题目辅导
            </button>
            <button className="send-action" disabled={store.sending || !draft.trim()} onClick={() => sendMessage()}>
              <Send size={17} />
              {store.sending ? "生成中" : "发送"}
            </button>
          </div>
        </footer>
      </main>

      {codeOpen ? (
        <div className="modal-backdrop" onMouseDown={() => setCodeOpen(false)}>
          <div className="code-modal" onMouseDown={(event) => event.stopPropagation()}>
            <h2 className="text-xl font-semibold text-stone-950">C 代码分析</h2>
            <input
              value={codeQuestion}
              onChange={(event) => setCodeQuestion(event.target.value)}
              placeholder="问题说明，可选"
            />
            <textarea
              value={codeValue}
              onChange={(event) => setCodeValue(event.target.value)}
              placeholder="#include <stdio.h>&#10;int main(void) {&#10;  return 0;&#10;}"
            />
            <div className="flex justify-end gap-2">
              <button className="ghost-action" onClick={() => setCodeOpen(false)}>取消</button>
              <button className="send-action" onClick={submitCodeAnalysis}>开始分析</button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function effectiveSubjectForMessage(payload: Record<string, unknown>, preferredSubject: SubjectId): SubjectId {
  const subjects = payload.subjects;
  if (Array.isArray(subjects) && subjects.length === 1 && subjects[0] === "C_program") {
    return "C_program";
  }
  return preferredSubject;
}

function isMessageDetails(value: unknown): value is MessageDetails {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
