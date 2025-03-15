'use client';

import { useState } from 'react';

interface CitationProps {
  sourceIndex: number;
  sources: any[];
}

export function Citation({ sourceIndex, sources }: CitationProps) {
  const [isHovering, setIsHovering] = useState(false);
  
  console.log(`Rendering Citation component for Source ${sourceIndex}`, { 
    sourcesLength: sources?.length || 0,
    sourceIndex,
    hasSource: sources && sourceIndex > 0 && sourceIndex <= sources.length,
    sourcesArray: JSON.stringify(sources?.map(s => ({ id: s.id, contentLength: s.content?.length || 0 })) || [])
  });
  
  // Find the source that matches this citation index
  let source = sources.find(s => {
    // Try both top-level citation_index and metadata.citation_index
    const topLevelIndex = s.citation_index;
    const metadataIndex = s.metadata?.citation_index;
    
    // Log for debugging
    console.log(`Checking source for citation ${sourceIndex}:`, {
      id: s.id,
      topLevelIndex,
      metadataIndex,
      content_preview: s.content?.substring(0, 50)
    });
    
    return topLevelIndex === sourceIndex || metadataIndex === sourceIndex;
  });
  
  // Handle missing source gracefully
  let sourceContent;
  if (!source || !source.content) {
    // Log warning for debugging
    console.warn(`Citation ${sourceIndex} not found in sources:`, {
      sourceIndex,
      availableSources: sources.map(s => ({
        id: s.id,
        topLevelIndex: s.citation_index,
        metadataIndex: s.metadata?.citation_index,
        content_preview: s.content?.substring(0, 50)
      }))
    });
    sourceContent = `Citation ${sourceIndex} not available`;
  } else {
    sourceContent = source.content;
  }

  console.log(`Source content for Source ${sourceIndex}:`, {
    source,
    hasContent: !!source?.content,
    contentPreview: source?.content ? source.content.substring(0, 50) + '...' : 'No content',
    sourceId: source?.id || 'No ID',
    sourceMetadata: source?.metadata || {}
  });
  
  // If only one source exists, override displayed index to 1
  const displayedCitationIndex = (safeSources => safeSources)(sources).length === 1 && sourceIndex !== 1 ? 1 : sourceIndex;
  
  return (
    <span 
      className="relative inline-block group"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <span className="cursor-pointer text-gray-700 hover:text-gray-900 font-medium underline decoration-dotted">
        [source {displayedCitationIndex}]
      </span>
      {isHovering && (
        <div className="absolute bottom-full left-0 w-80 bg-white border border-gray-200 text-gray-800 p-2 rounded-md shadow-sm 
                       z-50 text-sm mb-2">
          <pre className="whitespace-pre-wrap text-xs">{sourceContent}</pre>
        </div>
      )}
    </span>
  );
}
