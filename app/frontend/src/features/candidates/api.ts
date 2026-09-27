import { apiClient } from '../../shared/api/client'
import type { PageQuery, PageResponse } from '../../shared/api/types'
import type { Candidate, CandidateInput } from './types'

export async function getCandidates({ page = 1, pageSize = 20, query = '' }: PageQuery = {}): Promise<PageResponse<Candidate>> {
  return (await apiClient.get<PageResponse<Candidate>>('/candidates', {
    params: { page, page_size: pageSize, ...(query.trim() ? { q: query.trim() } : {}) },
  })).data
}

export async function createCandidate(payload: CandidateInput): Promise<Candidate> {
  return (await apiClient.post<Candidate>('/candidates', payload)).data
}

export async function updateCandidate(
  id: string,
  payload: Partial<CandidateInput>,
): Promise<Candidate> {
  return (await apiClient.patch<Candidate>(`/candidates/${id}`, payload)).data
}
