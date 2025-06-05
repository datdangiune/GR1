
export interface ChatMessage {
  id: string;
  message: string;
  isBot: boolean;
  timestamp: string;
}

export interface ChatState {
  messages: ChatMessage[];
  isTyping: boolean;
  error: string | null;
  selectedTopic: string;
}
