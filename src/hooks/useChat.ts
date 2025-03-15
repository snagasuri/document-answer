'use client';

import { useState, useEffect, useCallback } from 'react';
import { useApiClient } from '@/lib/api-client';
import { Message } from '@/types/chat';

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [tokenUsage, setTokenUsage] = useState<{
    promptTokens: number;
    completionTokens: number;
  }>({ promptTokens: 0, completionTokens: 0 });
  const [contextWindow, setContextWindow] = useState<{
    usedTokens: number;
    maxTokens: number;
  }>({ usedTokens: 0, maxTokens: 128000 });

  const { 
    getChatMessages, 
    streamChatResponse 
  } = useApiClient();

  // Load initial messages
  useEffect(() => {
    let isMounted = true;
    let isLoading = false;
    
    const loadMessages = async () => {
      if (isLoading) return;
      
      try {
        isLoading = true;
        setIsLoading(true);
        
        const messages = await getChatMessages(sessionId);
        
        if (isMounted) {
          setMessages(messages);
        }
      } catch (error) {
        if (isMounted) {
          console.error('Failed to load messages:', error);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
        isLoading = false;
      }
    };
    
    loadMessages();
    
    return () => {
      isMounted = false;
    };
  }, [sessionId]);

  const sendMessage = useCallback(async (message: string) => {
    try {
      setIsLoading(true);
      
      // Add user message immediately
      setMessages(prev => [
        ...prev,
        { 
          id: Date.now().toString(),
          role: 'user',
          content: message,
          timestamp: new Date().toISOString()
        }
      ]);

            // Stream assistant response
            await streamChatResponse(
                sessionId,
                message,
                (data) => {
                    setMessages(prev => {
                        const lastMessage = prev[prev.length - 1];
                        
                        // Convert sources from API format to our Source type
                        const newSources = data.sources ? data.sources.map((source: any, idx: number) => {
                            console.log('Processing source:', source);
                            // Ensure source has an id
                            if (!source.id) {
                                console.warn('Source missing ID, generating one');
                            }
                            
                            // Extract citation index from source metadata if available
                            let citationIndex = source.metadata?.citation_index;
                            
                            // If no citation index in metadata, try to extract from content
                            if (!citationIndex && source.content) {
                                // Look for patterns like "SOURCE 1" or "Source 2" in the content
                                const sourceMatch = source.content.match(/source\s+(\d+)/i);
                                if (sourceMatch) {
                                    citationIndex = parseInt(sourceMatch[1]);
                                }
                            }
                            
                            // Create enhanced metadata with source index information
                            const enhancedMetadata = {
                                ...(source.metadata || {}),
                                // Store the original array index (0-based)
                                array_index: idx,
                                // Store the citation index (1-based) if available, otherwise use array index + 1
                                index: citationIndex || (idx + 1)
                            };
                            
                            return {
                                id: source.id || `source-${Date.now()}`,
                                content: source.content || '',
                                metadata: enhancedMetadata
                            };
                        }) : [];
                        
                        // Merge new sources with existing sources, avoiding duplicates
                        let mergedSources = lastMessage.role === 'assistant' && lastMessage.sources 
                            ? [...lastMessage.sources] 
                            : [];
                            
                        // Add new sources that don't already exist in the merged sources
                        if (newSources.length > 0) {
                            newSources.forEach((newSource: any) => {
                                // Check if this source already exists in mergedSources
                                const existingSourceIndex = mergedSources.findIndex(
                                    (s: any) => s.id === newSource.id
                                );
                                
                                if (existingSourceIndex === -1) {
                                    // Source doesn't exist, add it
                                    mergedSources.push(newSource);
                                }
                            });
                        }
                        
                        // Log detailed source information
                        if (mergedSources.length > 0) {
                            console.log('Source details:');
                            mergedSources.forEach((source: any, index: number) => {
                                console.log(`Source ${index + 1}:`, {
                                    id: source.id,
                                    contentLength: source.content?.length || 0,
                                    metadata: source.metadata
                                });
                            });
                            
                            console.log(`Processed ${mergedSources.length} total sources`);
                            console.log('First source preview:', mergedSources[0]?.content?.substring(0, 50) + '...');
                        } else {
                            console.log('No sources in this chunk');
                            
                            // Check if there are citations in the content but no sources
                            const citationPattern = /\[Source (\d+)\]/g;
                            const content = data.content || '';
                            const matches = [...content.matchAll(citationPattern)];
                            if (matches.length > 0) {
                                console.log(`Warning: Found ${matches.length} citations but no sources provided`);
                                console.log('Citations found:', matches.map(m => m[0]).join(', '));
                            }
                        }
                        
                        if (lastMessage.role === 'assistant') {
                            // For streaming, we need to preserve the raw content for citation formatting
                            // Don't append to content directly, as it would break the React components
                            return [
                                ...prev.slice(0, -1),
                                {
                                    ...lastMessage,
                                    content: lastMessage.content + (data.content || ''),
                                    sources: mergedSources.length > 0 ? mergedSources : lastMessage.sources,
                                    // Store a flag to indicate this message is still streaming
                                    isStreaming: !data.finished
                                }
                            ];
                        }
            
            return [
              ...prev,
              {
                id: Date.now().toString(),
                role: 'assistant',
                content: data.content,
                timestamp: new Date().toISOString(),
                sources: newSources,
                isStreaming: !data.finished // Set initial streaming state
              }
            ];
          });

          // If this is the final message, update the message to mark it as not streaming
          if (data.finished) {
            setMessages(prev => {
              const lastMessage = prev[prev.length - 1];
              if (lastMessage.role === 'assistant') {
                console.log('Marking message as not streaming');
                return [
                  ...prev.slice(0, -1),
                  {
                    ...lastMessage,
                    isStreaming: false
                  }
                ];
              }
              return prev;
            });
          }

          // Always update token usage and context window on each chunk, even if values haven't changed
          // This ensures the UI stays in sync with the latest data
          
          // Handle token usage updates
          if (data.tokenUsage) {
            // Handle NaN values in token usage
            const promptTokens = isNaN(data.tokenUsage.prompt_tokens) ? 0 : data.tokenUsage.prompt_tokens;
            const completionTokens = isNaN(data.tokenUsage.completion_tokens) ? 0 : data.tokenUsage.completion_tokens;
            
            setTokenUsage({
              promptTokens,
              completionTokens
            });
            
            console.log('Token usage updated:', { promptTokens, completionTokens });
          }

          // Handle context window updates
          if (data.contextWindow) {
            // Handle NaN values in context window
            const usedTokens = isNaN(data.contextWindow.used_tokens) ? 0 : data.contextWindow.used_tokens;
            const maxTokens = isNaN(data.contextWindow.max_tokens) ? 100000 : data.contextWindow.max_tokens;
            
            setContextWindow({
              usedTokens,
              maxTokens
            });
            
            console.log('Context window updated:', { usedTokens, maxTokens });
          } else if (data.token_usage) {
            // Handle alternative token usage format (some API responses use different keys)
            const promptTokens = isNaN(data.token_usage.prompt_tokens) ? 0 : data.token_usage.prompt_tokens;
            const completionTokens = isNaN(data.token_usage.completion_tokens) ? 0 : data.token_usage.completion_tokens;
            const totalTokens = promptTokens + completionTokens;
            
            setTokenUsage({
              promptTokens,
              completionTokens
            });
            
            // Update context window based on token usage if contextWindow not provided
            setContextWindow(prev => ({
              usedTokens: totalTokens,
              maxTokens: prev.maxTokens
            }));
            
            console.log('Token usage (alternative format) updated:', { promptTokens, completionTokens, totalTokens });
          }
        },
        (error) => {
          console.error('Stream error:', error);
        },
        () => {
          setIsLoading(false);
        }
      );
    } catch (error) {
      console.error('Failed to send message:', error);
      setIsLoading(false);
    }
  }, [sessionId, streamChatResponse]);

  return {
    messages,
    sendMessage,
    isLoading,
    tokenUsage,
    contextWindow
  };
}
