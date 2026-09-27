export interface PageResponse<T> {
  items: T[]
  page: number
  page_size: number
  total: number
}

export interface PageQuery {
  page?: number
  pageSize?: number
  query?: string
}
