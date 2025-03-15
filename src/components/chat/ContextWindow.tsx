'use client';

interface ContextWindowProps {
  usedTokens: number;
  maxTokens: number;
  promptTokens: number;
  completionTokens: number;
}

export function ContextWindow({
  usedTokens,
  maxTokens,
  promptTokens,
  completionTokens
}: ContextWindowProps) {
  // Handle NaN values
  const safeUsedTokens = isNaN(usedTokens) ? 0 : usedTokens;
  const safeMaxTokens = isNaN(maxTokens) ? 128000 : maxTokens;
  const safePromptTokens = isNaN(promptTokens) ? 0 : promptTokens;
  const safeCompletionTokens = isNaN(completionTokens) ? 0 : completionTokens;
  
  const percentUsed = Math.min(100, Math.round((safeUsedTokens / safeMaxTokens) * 100));
  const remainingTokens = safeMaxTokens - safeUsedTokens;

  return (
    <div className="p-3 border-t border-gray-200 bg-gray-50">
      <div className="flex items-center justify-between text-xs text-gray-500 mb-2">
        <span>context window: {safeUsedTokens.toLocaleString()} / {safeMaxTokens.toLocaleString()} tokens</span>
        <span>{remainingTokens.toLocaleString()} remaining</span>
      </div>
      <div className="bg-gray-200 h-1.5 rounded-full">
        <div
          className="h-1.5 rounded-full bg-gray-500"
          style={{ width: `${percentUsed}%` }}
        />
      </div>
      <div className="flex items-center justify-between text-xs text-gray-500 mt-2">
        <span>↑ {safePromptTokens.toLocaleString()} input tokens</span>
        <span>↓ {safeCompletionTokens.toLocaleString()} output tokens</span>
      </div>
    </div>
  );
}
