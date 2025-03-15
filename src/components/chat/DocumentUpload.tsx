'use client';

interface DocumentUploadProps {
  sessionId: string;
  onUploadComplete?: (documentId: string, filename: string) => void;
}

export default function DocumentUpload({ sessionId, onUploadComplete }: DocumentUploadProps) {
  return (
    <div className="mb-3 p-3 bg-white border border-gray-200 rounded-md shadow-[0_1px_2px_rgba(0,0,0,0.03)]">
      <h3 className="text-sm font-medium text-gray-700 mb-2">upload document</h3>
      <p className="text-xs text-gray-500">
        click the document icon in the chat input to upload a pdf file.
      </p>
    </div>
  );
}
