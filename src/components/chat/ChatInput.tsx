'use client';

import { useState, useEffect } from 'react';
import { useApiClient } from '@/lib/api-client';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  documentsProcessing?: boolean;
  sessionId?: string;
}

export function ChatInput({ onSend, isLoading, documentsProcessing = false, sessionId }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const [uploadingDocument, setUploadingDocument] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [documentStatus, setDocumentStatus] = useState<string | null>(null);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const { uploadDocument, getDocumentStatus, refreshDocuments } = useApiClient();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSend(message);
      setMessage('');
    }
  };

  // Poll for document status if we have a document being processed
  useEffect(() => {
    if (!documentId || !sessionId) return;
    
    const checkStatus = async () => {
      try {
        const statusData = await getDocumentStatus(documentId);
        setDocumentStatus(statusData.processingStatus);
        
        if (statusData.processingStatus === 'complete') {
          // Document is ready, refresh the document list
          try {
            await refreshDocuments(sessionId);
            // Clear the document ID since processing is complete
            setDocumentId(null);
            setDocumentStatus(null);
          } catch (error) {
            console.error('Error refreshing documents:', error);
          }
        } else if (
          statusData.processingStatus === 'error' || 
          statusData.processingStatus === 'indexing_failed'
        ) {
          // Document processing failed
          setDocumentId(null);
        }
      } catch (error) {
        console.error('Error checking document status:', error);
        setDocumentId(null);
      }
    };
    
    // Check immediately
    checkStatus();
    
    // Then set up interval to check every 5 seconds
    const interval = setInterval(checkStatus, 5000);
    
    return () => clearInterval(interval);
  }, [documentId, sessionId, getDocumentStatus, refreshDocuments]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!sessionId) return;
    
    const file = event.target.files?.[0];
    if (!file) return;

    // Only accept PDF files
    if (file.type !== 'application/pdf') {
      alert('Only PDF files are supported');
      return;
    }

    setUploadingDocument(true);
    setUploadProgress(0);

    try {
      // Upload file using API client
      const data = await uploadDocument(file, sessionId);
      
      // Store the document ID for status polling
      setDocumentId(data.document_id);
      setDocumentStatus('processing');
      
      // Reset file input
      event.target.value = '';
    } catch (error) {
      console.error('Error uploading document:', error);
      alert(error instanceof Error ? error.message : 'Failed to upload document');
    } finally {
      setUploadingDocument(false);
    }
  };

  // Determine the button text and input placeholder based on document status
  const getButtonText = () => {
    if (uploadingDocument) return "Uploading...";
    if (documentStatus === 'processing') return "Processing...";
    if (documentStatus === 'indexed_not_searchable') return "Indexing...";
    if (documentsProcessing) return "Processing...";
    return "Send";
  };

  const getPlaceholder = () => {
    if (uploadingDocument) return "Uploading document...";
    if (documentStatus === 'processing') return "Document processing...";
    if (documentStatus === 'indexed_not_searchable') return "Document indexing...";
    if (documentsProcessing) return "Please wait for documents to finish processing...";
    return "Type your message...";
  };

  const isInputDisabled = isLoading || documentsProcessing || uploadingDocument || 
    documentStatus === 'processing' || documentStatus === 'indexed_not_searchable';

  return (
    <form onSubmit={handleSubmit} className="chat-input-form">
      {sessionId && (
        <label className="chat-input-file-label">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="chat-input-file-icon" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={1.5} 
              d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" 
            />
          </svg>
          <input
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={handleFileUpload}
            disabled={isInputDisabled}
          />
        </label>
      )}
      
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder={getPlaceholder()}
        className="chat-input"
        disabled={isInputDisabled}
      />
      <button
        type="submit"
        disabled={isInputDisabled}
        className="chat-send-button"
      >
        {getButtonText()}
      </button>
    </form>
  );
}
