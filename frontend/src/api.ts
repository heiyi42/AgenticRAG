import type { ChatSession, GraphPayload, RetrievedChunk, StreamEvent } from "./types";

async function apiJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof data?.error === "string" ? data.error : `HTTP ${response.status}`;
    throw new Error(message);
  }
  return data as T;
}

export const api = {
  listChats: () => apiJson<{ chats: ChatSession[] }>("/api/chats"),
  createChat: (mode: string) =>
    apiJson<ChatSession>("/api/chats", {
      method: "POST",
      body: JSON.stringify({ mode })
    }),
  getChat: (chatId: string) => apiJson<ChatSession>(`/api/chats/${chatId}`),
  deleteChat: (chatId: string) =>
    apiJson<{ ok: boolean }>(`/api/chats/${chatId}/delete`, { method: "POST" }),
  renameChat: (chatId: string, title: string) =>
    apiJson<ChatSession>(`/api/chats/${chatId}/rename`, {
      method: "POST",
      body: JSON.stringify({ title })
    }),
  pinChat: (chatId: string, pinned: boolean) =>
    apiJson<ChatSession>(`/api/chats/${chatId}/pin`, {
      method: "POST",
      body: JSON.stringify({ pinned })
    }),
  setMode: (chatId: string, mode: string) =>
    apiJson<ChatSession>(`/api/chats/${chatId}/mode`, {
      method: "POST",
      body: JSON.stringify({ mode })
    }),
  graphHealth: () => apiJson<Record<string, unknown>>("/api/graph/health"),
  localSubgraph: (payload: {
    subjectIds: string[];
    query: string;
    centerEntityIds?: string[];
    depth?: number;
    limit?: number;
  }) =>
    apiJson<GraphPayload>("/api/graph/local-subgraph", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  entityChunks: (entityId: string) =>
    apiJson<{ ok: boolean; chunks: RetrievedChunk[] }>(
      `/api/graph/entity/${encodeURIComponent(entityId)}/chunks`
    )
};

export async function streamChatMessage(
  chatId: string,
  payload: Record<string, unknown>,
  signal: AbortSignal,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  const response = await fetch(`/api/chats/${chatId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal
  });
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data?.error === "string" ? data.error : `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const parsed = parseSseBlock(part);
      if (parsed) onEvent(parsed);
    }
  }
}

function parseSseBlock(block: string): StreamEvent | null {
  const lines = block.split(/\r?\n/);
  let event = "message";
  let data = "";
  for (const line of lines) {
    if (line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      data += line.slice("data:".length).trim();
    }
  }
  if (!data) return null;
  try {
    return { event, data: JSON.parse(data) } as StreamEvent;
  } catch {
    return { event, data: { text: data } } as StreamEvent;
  }
}
