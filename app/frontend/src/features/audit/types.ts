export interface AuditLogEntry {
  id: string
  actor_user_id: string | null
  actor_username: string | null
  actor_full_name: string | null
  action: string
  entity_type: string
  entity_id: string | null
  metadata: Record<string, unknown> | null
  created_at: string
}
