export interface ContentRect {
  x: number
  y: number
  width: number
  height: number
}

export function containedVideoRect(
  containerWidth: number,
  containerHeight: number,
  sourceWidth: number,
  sourceHeight: number,
): ContentRect {
  if (containerWidth <= 0 || containerHeight <= 0 || sourceWidth <= 0 || sourceHeight <= 0) {
    return { x: 0, y: 0, width: 0, height: 0 }
  }
  const sourceAspect = sourceWidth / sourceHeight
  const containerAspect = containerWidth / containerHeight
  const width = sourceAspect >= containerAspect ? containerWidth : containerHeight * sourceAspect
  const height = sourceAspect >= containerAspect ? containerWidth / sourceAspect : containerHeight
  return { x: (containerWidth - width) / 2, y: (containerHeight - height) / 2, width, height }
}

export function mapNormalizedBox(
  box: [number, number, number, number],
  rect: ContentRect,
): [number, number, number, number] {
  const [x1, y1, x2, y2] = box
  return [
    rect.x + x1 * rect.width,
    rect.y + y1 * rect.height,
    rect.x + x2 * rect.width,
    rect.y + y2 * rect.height,
  ]
}
