'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useApiClient } from '@/lib/api-client';

interface Document {
  _id: string;
  filename: string;
  createdAt: string;
  processingStatus?: string;
  metadata: {
    pages?: number;
    title?: string;
    author?: string;
    [key: string]: any;
  };
}

interface DocumentListProps {
  sessionId: string;
  onDeleteDocument?: (documentId: string) => void;
  onProcessingStatusChange?: (isProcessing: boolean) => void;
}

export default function DocumentList({ sessionId, onDeleteDocument, onProcessingStatusChange }: DocumentListProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { getSessionDocuments, deleteDocument } = useApiClient();

  // Fetch documents for the session
  const fetchDocuments = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await getSessionDocuments(sessionId);
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch documents');
    } finally {
      setIsLoading(false);
    }
  };

  // Delete a document
  const handleDeleteDocument = async (documentId: string) => {
    try {
      await deleteDocument(documentId, sessionId);

      // Remove document from state
      setDocuments(documents.filter(doc => doc._id !== documentId));

      // Call onDeleteDocument callback
      if (onDeleteDocument) {
        onDeleteDocument(documentId);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      setError(error instanceof Error ? error.message : 'Failed to delete document');
    }
  };

  // Check for processing documents and notify parent component
  useEffect(() => {
    const hasProcessingDocuments = documents.some(doc => 
      doc.processingStatus === 'processing' || 
      doc.processingStatus === 'indexed_not_searchable'
    );
    
    if (onProcessingStatusChange) {
      onProcessingStatusChange(hasProcessingDocuments);
    }
  }, [documents, onProcessingStatusChange]);

  // Fetch documents on mount and when sessionId changes
  useEffect(() => {
    if (sessionId) {
      fetchDocuments();
    }
  }, [sessionId]);

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  if (isLoading) {
    return (
      <div className="p-3 macos-card mb-3">
        <h3 className="macos-title mb-2">documents</h3>
        <p className="macos-caption">loading documents...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-3 macos-card mb-3">
        <h3 className="macos-title mb-2">documents</h3>
        <div className="p-2 bg-gray-100 text-gray-700 rounded-md">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="p-3 macos-card mb-3">
      <h3 className="macos-title mb-2">documents</h3>
      
      {documents.length === 0 ? (
        <p className="macos-caption">
          no documents uploaded yet. upload a document to start chatting with it.
        </p>
      ) : (
        <ul className="divide-y divide-gray-200">
          {documents.map((doc) => (
            <li key={doc._id} className="py-2">
              <div className="flex justify-between">
              <div>
                <h4 className="macos-subtitle">{doc.filename}</h4>
                <p className="macos-caption">
                  uploaded: {formatDate(doc.createdAt)}
                </p>
                {doc.metadata.pages && (
                  <p className="macos-caption">
                    pages: {doc.metadata.pages}
                  </p>
                )}
                {doc.processingStatus && (
                  <p className={`macos-caption ${
                    doc.processingStatus === 'complete' ? 'text-gray-700' : 
                    doc.processingStatus === 'error' ? 'text-gray-700' : 
                    doc.processingStatus === 'indexing_failed' ? 'text-gray-700' : 
                    'text-gray-500'}`}>
                    status: {
                      doc.processingStatus === 'processing' ? 'processing...' : 
                      doc.processingStatus === 'complete' ? 'ready' : 
                      doc.processingStatus === 'indexed_not_searchable' ? 'indexing...' : 
                      doc.processingStatus === 'indexing_failed' ? 'indexing failed' : 
                      'error'
                    }
                  </p>
                )}
              </div>
                <button
                  onClick={() => handleDeleteDocument(doc._id)}
                  className="text-gray-500 hover:text-gray-700"
                  title="delete document"
                >
                  <svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    className="h-5 w-5" 
                    fill="none" 
                    viewBox="0 0 24 24" 
                    stroke="currentColor"
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={1.5} 
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" 
                    />
                  </svg>
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
