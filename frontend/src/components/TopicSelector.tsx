
import React from 'react';
import { ChevronDown } from 'lucide-react';

interface TopicSelectorProps {
  selectedTopic: string;
  onTopicChange: (topic: string) => void;
  isDark: boolean;
}

const topics = [
  { value: 'general', label: 'General Health' },
  { value: 'cardiology', label: 'Cardiology' },
  { value: 'digestion', label: 'Digestion' },
  { value: 'respiratory', label: 'Respiratory' },
  { value: 'nutrition', label: 'Nutrition' },
  { value: 'mental-health', label: 'Mental Health' }
];

const TopicSelector: React.FC<TopicSelectorProps> = ({ selectedTopic, onTopicChange, isDark }) => {
  return (
    <div className="relative mb-4">
      <select
        value={selectedTopic}
        onChange={(e) => onTopicChange(e.target.value)}
        className={`w-full px-3 py-2 rounded-lg border appearance-none cursor-pointer ${
          isDark 
            ? 'bg-gray-700 border-gray-600 text-white' 
            : 'bg-white border-gray-300 text-gray-700'
        } focus:outline-none focus:ring-2 focus:ring-blue-500`}
      >
        {topics.map((topic) => (
          <option key={topic.value} value={topic.value}>
            {topic.label}
          </option>
        ))}
      </select>
      <ChevronDown className={`absolute right-3 top-3 w-4 h-4 pointer-events-none ${
        isDark ? 'text-gray-400' : 'text-gray-500'
      }`} />
    </div>
  );
};

export default TopicSelector;
