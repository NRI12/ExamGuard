import type { ReactNode } from 'react'

export function FormField({ label, htmlFor, required = false, error, helper, children }: {
  label: string
  htmlFor: string
  required?: boolean
  error?: string
  helper?: string
  children: ReactNode
}) {
  const messageId = `${htmlFor}-message`
  return <div className={`form-field ${error ? 'invalid' : ''}`}>
    <label htmlFor={htmlFor}>{label}{required && <span aria-hidden="true"> *</span>}</label>
    {children}
    {(error || helper) && <small id={messageId} className={error ? 'field-error' : 'field-helper'}>{error ?? helper}</small>}
  </div>
}
