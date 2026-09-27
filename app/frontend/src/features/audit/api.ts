import { apiClient } from '../../shared/api/client'
import type { PageResponse } from '../../shared/api/types'
import type { AuditLogEntry } from './types'

export async function getAuditLogs(page = 1, pageSize = 20): Promise<PageResponse<AuditLogEntry>> {
  return (await apiClient.get<PageResponse<AuditLogEntry>>('/audit-logs', {
    params: { page, page_size: pageSize },
  })).data
}
