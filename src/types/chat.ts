export interface Source {
  id: string;
  content: string;
  metadata?: any;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: Source[];
  isStreaming?: boolean;
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
}

export interface ContextWindow {
  usedTokens: number;
  maxTokens: number;
}
