import { BarChart } from 'echarts/charts'
import { GridComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { SVGRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'

echarts.use([BarChart, GridComponent, TooltipComponent, MarkAreaComponent, MarkLineComponent, SVGRenderer])

export type ChartOption = echarts.EChartsCoreOption

interface EChartProps {
  option: ChartOption
  /** Charts of one group share hover tooltips (comparison slots). */
  group?: string
  onBarClick?: (index: number) => void
  onWidth?: (width: number) => void
  label: string
}

/**
 * SVG rendering stays crisp at any browser zoom; animation is off because
 * data changes are instant state changes, not a show.
 */
export function EChart({ option, group, onBarClick, onWidth, label }: EChartProps) {
  const element = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  const clickRef = useRef(onBarClick)
  const widthRef = useRef(onWidth)
  clickRef.current = onBarClick
  widthRef.current = onWidth

  useEffect(() => {
    if (!element.current) return
    const instance = echarts.init(element.current, undefined, { renderer: 'svg' })
    chart.current = instance
    instance.on('click', (params) => {
      if (params.componentType === 'series' && typeof params.dataIndex === 'number') clickRef.current?.(params.dataIndex)
    })
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? 0
      widthRef.current?.(width)
      instance.resize()
    })
    observer.observe(element.current)
    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [])

  useEffect(() => {
    chart.current?.setOption({ animation: false, ...option }, { notMerge: true })
  }, [option])

  useEffect(() => {
    const instance = chart.current
    if (!instance || !group) return
    instance.group = group
    echarts.connect(group)
    return () => {
      instance.group = ''
    }
  }, [group])

  return <div ref={element} role="img" aria-label={label} style={{ width: '100%', height: '100%' }} />
}
