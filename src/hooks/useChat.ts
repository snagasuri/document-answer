'use client';

import { useState, useEffect, useCallback } from 'react';
import { useApiClient } from '@/lib';
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
                        
                        interface Source {
                          id: string;
                          content: string;
                          citation_index: number;
                          metadata: {
                            citation_index: number;
                            [key: string]: any;
                          };
                        }

                        // Process sources with properly indexed citations
            const newSources = data.sources ? data.sources.map((source: any, index: number): Source => {
                // Ensure a unique source ID
                const sourceId = source.id || `source-${Date.now()}-${Math.random()}`;
                
                // Get citation index from metadata or calculate from array position
                const citationIndex = source.metadata?.citation_index || index + 1;
                
                // Create new metadata object with citation_index
                const metadata = {
                    ...(source.metadata || {}),
                    citation_index: citationIndex
                };

                return {
                    id: sourceId,
                    content: source.content || '',
                    citation_index: citationIndex,  // Set both in metadata and top level
                    metadata
                };
            }) : [];
                        
                        // Update sources for message, preserving existing sources
                        const existingSources = lastMessage?.sources || [];
                        const sources = [...existingSources, ...newSources];
                        
                        // Log source processing for debugging
                        if (sources) {
                            console.log('Processed sources:', sources.map((s: Source) => ({
                                id: s.id,
                                citation_index: s.metadata.citation_index
                            })));
                        }
                        
                        if (lastMessage.role === 'assistant') {
                            // For streaming, we need to preserve the raw content for citation formatting
                            // Don't append to content directly, as it would break the React components
                            return [
                                ...prev.slice(0, -1),
                                {
                                    ...lastMessage,
                                    content: lastMessage.content + (data.content || ''),
                                    sources: lastMessage.sources ? [...lastMessage.sources, ...newSources] : newSources,
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
                sources: lastMessage?.sources ? [...lastMessage.sources, ...newSources] : newSources,
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
