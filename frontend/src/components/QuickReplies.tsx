
import React from 'react';

interface QuickRepliesProps {
  onQuickReply: (message: string) => void;
  isDark: boolean;
}

const quickReplies = [
  "I have a headache",
  "I need a nutrition tip",
  "I'm feeling anxious",
  "I have chest pain",
  "I need exercise advice",
  "I have digestive issues"
];

const QuickReplies: React.FC<QuickRepliesProps> = ({ onQuickReply, isDark }) => {
  return (
    <div className="mb-4">
      <p className={`text-sm mb-2 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
        Quick replies:
      </p>
      <div className="flex flex-wrap gap-2">
        {quickReplies.map((reply, index) => (
          <button
            key={index}
            onClick={() => onQuickReply(reply)}
            className={`px-3 py-1 text-sm rounded-full border transition-colors ${
              isDark
                ? 'border-gray-600 text-gray-300 hover:bg-gray-700'
                : 'border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            {reply}
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuickReplies;
