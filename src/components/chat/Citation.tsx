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
  
  // The sources array is 0-indexed, but citations are 1-indexed
  // We need to handle this mismatch and also handle the case where sources might be out of order
  
  // First, try to find a source with an index property matching sourceIndex
  let source = null;
  if (sources && sources.length > 0) {
    // Try to find a source with metadata.index === sourceIndex
    source = sources.find(s => s.metadata && s.metadata.index === sourceIndex);
    
    // If not found by index metadata, fall back to array position
    if (!source && sourceIndex > 0 && sourceIndex <= sources.length) {
      source = sources[sourceIndex-1];
    }
  }
  
  // Handle missing source gracefully
  if (!source) {
    console.warn(`Citation source ${sourceIndex} not found in sources array of length ${sources?.length || 0}`);
    
    // Check if we have any sources at all and log their details
    if (sources && sources.length > 0) {
      console.log('Available source indices:', sources.map((_, i) => i + 1).join(', '));
      console.log('Source metadata:', sources.map(s => ({
        id: s.id,
        index: s.metadata?.index,
        contentPreview: s.content?.substring(0, 30) + '...'
      })));
    }
  }
  
  // Create a more informative fallback message
  let sourceContent = "Source content not available";
  if (source) {
    sourceContent = source.content || "Source content exists but is empty";
  } else if (!sources || sources.length === 0) {
    sourceContent = "No sources available for this response";
  } else if (sourceIndex > sources.length) {
    sourceContent = `Source ${sourceIndex} not found. Only ${sources.length} sources are available.`;
    
    // Show the last available source as a fallback
    const lastSource = sources[sources.length - 1];
    if (lastSource && lastSource.content) {
      sourceContent += `\n\nShowing content from last available source instead:\n\n${lastSource.content}`;
    }
  }
  
  console.log(`Source content for Source ${sourceIndex}:`, {
    hasContent: !!source?.content,
    contentPreview: source?.content ? source.content.substring(0, 50) + '...' : 'No content',
    sourceId: source?.id || 'No ID',
    sourceMetadata: source?.metadata || {}
  });
  
  return (
    <span 
      className="relative inline-block"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <span className="cursor-pointer text-gray-700 hover:text-gray-900 font-medium">
        [source {sourceIndex}]
      </span>
      {isHovering && (
        <div className="absolute bottom-full left-0 w-80 bg-white border border-gray-200 text-gray-800 p-2 rounded-md shadow-macos 
                       z-50 text-sm max-h-60 overflow-y-auto mb-1">
          {sourceContent}
        </div>
      )}
    </span>
  );
}
