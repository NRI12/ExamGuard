export interface Candidate {
  id: string
  candidate_code: string
  full_name: string
  class_name: string | null
  note: string | null
  created_at: string
  updated_at: string
}

export type CandidateInput = Pick<
  Candidate,
  'candidate_code' | 'full_name' | 'class_name' | 'note'
>
