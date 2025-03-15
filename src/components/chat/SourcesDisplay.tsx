'use client';

interface SourcesDisplayProps {
  sources: any[];
}

export function SourcesDisplay({ sources }: SourcesDisplayProps) {
  if (!sources || sources.length === 0) return null;
  return (
    <div className="p-3 border-t border-gray-200 bg-gray-50 mt-2">
      <h4 className="text-sm font-medium text-gray-800 mb-2">Sources</h4>
      <ul className="text-xs text-gray-700 space-y-1">
        {sources.map((source, idx) => {
          const citationIndex = source.metadata?.citation_index || idx + 1;
          const contentPreview = source.content
            ? (source.content.length > 100
                ? source.content.substring(0, 100) + "..."
                : source.content)
            : "Content not available";
          return (
            <li key={source.id || idx}>
              <span className="font-medium">[Source {citationIndex}]:</span> {contentPreview}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
