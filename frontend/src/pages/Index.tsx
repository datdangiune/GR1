
import React from 'react';
import Header from '../components/Header';
import ChatMessage from '../components/ChatMessage';
import TypingIndicator from '../components/TypingIndicator';
import TopicSelector from '../components/TopicSelector';
import QuickReplies from '../components/QuickReplies';
import ChatInput from '../components/ChatInput';
import ErrorMessage from '../components/ErrorMessage';
import { useChat } from '../hooks/useChat';
import { useTheme } from '../hooks/useTheme';
import { Trash2 } from 'lucide-react';

const Index = () => {
  const {
    messages,
    isTyping,
    error,
    selectedTopic,
    sendMessage,
    clearConversation,
    setSelectedTopic,
    dismissError,
    chatEndRef
  } = useChat();

  const { isDark, toggleTheme } = useTheme();

  const handleQuickReply = (message: string) => {
    sendMessage(message);
  };

  return (
    <div className={`min-h-screen flex flex-col ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      <Header isDark={isDark} toggleTheme={toggleTheme} />
      
      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full">
        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="text-center py-12">
              <div className="bg-blue-500 w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center">
                <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clipRule="evenodd" />
                </svg>
              </div>
              <h2 className={`text-xl font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-800'}`}>
                Welcome to MediBot
              </h2>
              <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} max-w-md mx-auto`}>
                I'm here to provide general health information and guidance. Please remember that I cannot replace professional medical advice.
              </p>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  message={msg.message}
                  isBot={msg.isBot}
                  timestamp={msg.timestamp}
                  isDark={isDark}
                />
              ))}
              {isTyping && <TypingIndicator isDark={isDark} />}
            </>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Controls Area */}
        <div className={`border-t p-4 ${isDark ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'}`}>
          {error && (
            <ErrorMessage 
              message={error}
              onDismiss={dismissError}
              isDark={isDark}
            />
          )}
          
          <div className="flex items-center justify-between mb-4">
            <TopicSelector
              selectedTopic={selectedTopic}
              onTopicChange={setSelectedTopic}
              isDark={isDark}
            />
            
            {messages.length > 0 && (
              <button
                onClick={clearConversation}
                className={`ml-4 px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-2 ${
                  isDark
                    ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <Trash2 className="w-4 h-4" />
                <span>Clear</span>
              </button>
            )}
          </div>
          
          <QuickReplies onQuickReply={handleQuickReply} isDark={isDark} />
        </div>

        {/* Input Area */}
        <ChatInput
          onSendMessage={sendMessage}
          disabled={isTyping}
          isDark={isDark}
        />
      </div>
    </div>
  );
};

export default Index;
