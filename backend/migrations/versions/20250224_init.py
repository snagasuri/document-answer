"""Initial migration

Revision ID: 20250224_init
Revises: 
Create Date: 2025-02-24 20:22:28.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20250224_init'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('doc_metadata', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
    )
    
    # Create document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_metadata', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('vector_id', sa.String()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    )
    
    # Create chat_sessions table
    op.create_table(
        'chat_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('session_metadata', postgresql.JSONB(), nullable=False, server_default='{}'),
    )
    
    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('message_metadata', postgresql.JSONB(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ondelete='CASCADE'),
    )
    
    # Create indexes
    op.create_index('idx_documents_created_at', 'documents', ['created_at'])
    op.create_index('idx_chunks_document_id', 'document_chunks', ['document_id'])
    op.create_index('idx_chunks_vector_id', 'document_chunks', ['vector_id'])
    op.create_index('idx_messages_session_id', 'chat_messages', ['session_id'])
    op.create_index('idx_messages_created_at', 'chat_messages', ['created_at'])
    
    # Create GIN indexes for JSONB columns
    op.execute(
        'CREATE INDEX idx_documents_metadata ON documents USING GIN (doc_metadata jsonb_path_ops)'
    )
    op.execute(
        'CREATE INDEX idx_chunks_metadata ON document_chunks USING GIN (chunk_metadata jsonb_path_ops)'
    )
    op.execute(
        'CREATE INDEX idx_sessions_metadata ON chat_sessions USING GIN (session_metadata jsonb_path_ops)'
    )
    op.execute(
        'CREATE INDEX idx_messages_metadata ON chat_messages USING GIN (message_metadata jsonb_path_ops)'
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('chat_messages')
    op.drop_table('chat_sessions')
    op.drop_table('document_chunks')
    op.drop_table('documents')
