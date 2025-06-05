
import React from 'react';
import { Bot } from 'lucide-react';

interface TypingIndicatorProps {
  isDark: boolean;
}

const TypingIndicator: React.FC<TypingIndicatorProps> = ({ isDark }) => {
  return (
    <div className="flex justify-start mb-4">
      <div className="flex">
        <div className="flex-shrink-0 mr-3">
          <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
            <Bot className="w-5 h-5 text-white" />
          </div>
        </div>
        
        <div className={`rounded-lg px-4 py-2 shadow-md ${
          isDark ? 'bg-gray-700' : 'bg-gray-100'
        }`}>
          <div className="flex space-x-1">
            <div className={`w-2 h-2 rounded-full animate-pulse ${
              isDark ? 'bg-gray-400' : 'bg-gray-500'
            }`} style={{ animationDelay: '0ms' }}></div>
            <div className={`w-2 h-2 rounded-full animate-pulse ${
              isDark ? 'bg-gray-400' : 'bg-gray-500'
            }`} style={{ animationDelay: '150ms' }}></div>
            <div className={`w-2 h-2 rounded-full animate-pulse ${
              isDark ? 'bg-gray-400' : 'bg-gray-500'
            }`} style={{ animationDelay: '300ms' }}></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
