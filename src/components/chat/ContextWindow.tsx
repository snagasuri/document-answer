'use client';

interface ContextWindowProps {
  usedTokens: number;
  maxTokens: number;
  promptTokens: number;
  completionTokens: number;
  sources?: any[]; // Add sources
  activeCitation?: number | null; // Currently active citation
}

export function ContextWindow({
  usedTokens,
  maxTokens,
  promptTokens,
  completionTokens,
  sources = [],
  activeCitation = null
}: ContextWindowProps) {
  // Handle NaN values
  const safeUsedTokens = isNaN(usedTokens) ? 0 : usedTokens;
  const safeMaxTokens = isNaN(maxTokens) ? 128000 : maxTokens;
  const safePromptTokens = isNaN(promptTokens) ? 0 : promptTokens;
  const safeCompletionTokens = isNaN(completionTokens) ? 0 : completionTokens;
  
  const percentUsed = Math.min(100, Math.round((safeUsedTokens / safeMaxTokens) * 100));
  const remainingTokens = safeMaxTokens - safeUsedTokens;

  return (
    <div className="context-window">
      <div className="context-window-header">
        <span className="context-window-caption">context window: {safeUsedTokens.toLocaleString()} / {safeMaxTokens.toLocaleString()} tokens</span>
        <span className="context-window-caption">{remainingTokens.toLocaleString()} remaining</span>
      </div>
      <div className="context-window-progress-bg">
        <div
          className="context-window-progress-fill"
          style={{ width: `${percentUsed}%` }}
        />
      </div>
      <div className="context-window-footer">
        <span>↑ {safePromptTokens.toLocaleString()} input tokens</span>
        <span>↓ {safeCompletionTokens.toLocaleString()} output tokens</span>
      </div>
      
      {/* Add source display section */}
      {sources.length > 0 && (
        <div className="mt-3">
          <h3 className="macos-subtitle mb-1">sources</h3>
          <div className="max-h-60 overflow-y-auto border border-gray-200 rounded-md">
            {sources.map((source, arrayIndex) => {
              // Determine the display index for this source
              // Use metadata.index if available, otherwise use array index + 1
              const displayIndex = source.metadata?.index || (arrayIndex + 1);
              
              return (
                <div 
                  key={`source-${source.id || arrayIndex}`}
                  className={`p-2 border-b border-gray-200 text-xs ${
                    activeCitation === displayIndex ? 'bg-gray-100' : ''
                  }`}
                >
                  <div className="macos-subtitle">[source {displayIndex}]</div>
                  <div className="mt-1 whitespace-pre-wrap">{source.content}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
