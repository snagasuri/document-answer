import React from 'react';
import { Citation } from '../components/chat/Citation';

/**
 * Formats a message string by replacing citation patterns with interactive Citation components
 * 
 * @param message The message text containing citation patterns like [Source 1]
 * @param sources Array of source objects that contain the content to display in tooltips
 * @returns An array of React elements and strings representing the formatted message
 */
export function formatMessageWithCitations(message: string, sources: any[] = []) {
  if (!message) return [];
  
  // Ensure sources is always an array
  const safeSources = Array.isArray(sources) ? sources : [];
  
  console.log('Formatting message with citations:', { 
    messageLength: message.length,
    sourcesCount: safeSources.length,
    messagePreview: message.substring(0, 50) + '...',
    hasSources: safeSources.length > 0,
    firstSourceId: safeSources[0]?.id || 'No sources available'
  });
  
  // Log source metadata for debugging
  if (safeSources.length > 0) {
    console.log('Source metadata for citation mapping:', safeSources.map(s => ({
      id: s.id,
      index: s.metadata?.index,
      array_index: s.metadata?.array_index,
      contentPreview: s.content?.substring(0, 30) + '...'
    })));
  }
  
  // Regex to find citation patterns
  const citationPattern = /\[Source (\d+)\]/g;
  
  // Extract all citation indices from the message
  const citationIndices = new Set<number>();
  let citationMatch;
  while ((citationMatch = citationPattern.exec(message)) !== null) {
    citationIndices.add(parseInt(citationMatch[1]));
  }
  
  // Check if we have all the cited sources
  if (citationIndices.size > 0) {
    console.log('Citation indices found in message:', Array.from(citationIndices).join(', '));
    
    // Check if we're missing any sources
    const missingIndices = Array.from(citationIndices).filter(idx => 
      !safeSources.some(s => s.metadata?.index === idx)
    );
    
    if (missingIndices.length > 0) {
      console.warn(`Missing sources for indices: ${missingIndices.join(', ')}`);
    }
  }
  
  // Split message into parts and replace citations with components
  let lastIndex = 0;
  const parts: React.ReactNode[] = [];
  let match;
  let citationsFound = 0;
  
  // Reset the regex
  citationPattern.lastIndex = 0;
  
  while ((match = citationPattern.exec(message)) !== null) {
    citationsFound++;
    console.log(`Found citation: [Source ${match[1]}] at index ${match.index}`);
    
    // Add text before citation
    if (match.index > lastIndex) {
      parts.push(message.substring(lastIndex, match.index));
    }
    
    // Add citation component
    const sourceIndex = parseInt(match[1]);
    parts.push(
      <Citation 
        key={`citation-${match.index}`} 
        sourceIndex={sourceIndex} 
        sources={safeSources} 
      />
    );
    
    lastIndex = match.index + match[0].length;
  }
  
  // Add remaining text
  if (lastIndex < message.length) {
    parts.push(message.substring(lastIndex));
  }
  
  console.log(`Formatting complete: Found ${citationsFound} citations, created ${parts.length} parts`, {
    citationsFound,
    partsCount: parts.length,
    sourcesAvailable: safeSources.length
  });
  
  return parts;
}
