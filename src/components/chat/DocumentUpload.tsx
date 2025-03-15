'use client';

import { useApiClient } from '@/lib/api-client';

interface DocumentUploadProps {
  sessionId: string;
  onUploadComplete?: (documentId: string, filename: string) => void;
}

export default function DocumentUpload({ sessionId, onUploadComplete }: DocumentUploadProps) {
  return (
    <div className="mb-3 p-3 macos-card">
      <h3 className="macos-title mb-2">upload document</h3>
      <p className="macos-caption mb-2">
        you can upload pdf documents directly from the chat input box. 
        click the document icon in the chat input to upload a pdf.
      </p>
      <p className="macos-caption">
        the document will be automatically processed and you'll be notified when it's ready to use.
      </p>
    </div>
  );
}
