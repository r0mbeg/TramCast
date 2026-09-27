import { formatShort, inHorizon, weekday } from '../../lib/calendar'
import { formatNumber, hourLabel, hourRange, isPartialHour, isServiceOff } from '../../lib/format'
import { LOAD_THRESHOLDS, loadHex } from '../../lib/loadScale'
import { hex } from '../../lib/theme'
import type { ChartOption } from './EChart'

const FONT = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"

function escape(text: string): string {
  return text.replace(/[&<>"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[char] ?? char)
}

const tooltip = {
  trigger: 'axis',
  axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(255,255,255,0.04)' } },
  backgroundColor: hex.surface3,
  borderColor: hex.border,
  borderWidth: 1,
  padding: [6, 10],
  textStyle: { color: hex.text1, fontFamily: FONT, fontSize: 13 },
  extraCssText: 'box-shadow:none;border-radius:6px;',
}

function axisBase() {
  return {
    axisLine: { lineStyle: { color: hex.border } },
    axisTick: { show: false },
    axisLabel: { color: hex.text3, fontFamily: FONT, fontSize: 12 },
    splitLine: { show: false },
  }
}

/** Bars need about this many pixels for a readable number above them. */
const LABEL_MIN_BAR = 30

export interface DayChartInput {
  values: readonly number[]
  selectedHour: number
  yMax: number
  width: number
}

export function dayChartOption({ values, selectedHour, yMax, width }: DayChartInput): ChartOption {
  const peakHour = values.reduce((best, value, hour) => (value > (values[best] ?? 0) ? hour : best), 0)
  const allLabels = width / 24 >= LABEL_MIN_BAR
  return {
    textStyle: { fontFamily: FONT },
    grid: { left: 52, right: 56, top: 26, bottom: 28 },
    tooltip: {
      ...tooltip,
      formatter: (params: { dataIndex: number }[]) => {
        const hour = params[0]?.dataIndex ?? 0
        const value = values[hour] ?? 0
        const body = isServiceOff(hour) ? 'терминалы выключены, прогноз 0' : `${formatNumber(value)} посадок`
        const note = isPartialHour(hour) ? '<br/><span style="color:' + hex.text3 + '">учитываются посадки с 05:30</span>' : ''
        return `${hourRange(hour)} — ${escape(body)}${note}`
      },
    },
    xAxis: {
      type: 'category',
      data: values.map((_, hour) => (isPartialHour(hour) ? '05*' : hourLabel(hour))),
      ...axisBase(),
      axisLabel: {
        ...axisBase().axisLabel,
        interval: width < 520 ? 2 : 0,
        formatter: (label: string, index: number) => (index === selectedHour ? `{sel|${label}}` : label),
        rich: { sel: { color: hex.text1, fontWeight: 600, fontFamily: FONT, fontSize: 12 } },
      },
    },
    yAxis: {
      type: 'value',
      max: yMax,
      min: 0,
      ...axisBase(),
      axisLine: { show: false },
      axisLabel: { ...axisBase().axisLabel, formatter: (value: number) => formatNumber(value) },
      splitNumber: 4,
    },
    series: [
      {
        type: 'bar',
        barCategoryGap: '25%',
        data: values.map((value, hour) => {
          const off = isServiceOff(hour)
          const color = off ? hex.surface3 : (loadHex(value) ?? hex.surface3)
          return {
            value: off ? 0 : value,
            itemStyle: {
              color,
              borderRadius: [3, 3, 0, 0],
              borderColor: hour === selectedHour ? hex.selection : 'transparent',
              borderWidth: hour === selectedHour ? 2 : 0,
            },
          }
        }),
        label: {
          show: true,
          position: 'top',
          color: hex.text2,
          fontFamily: FONT,
          fontSize: 11,
          formatter: (params: { dataIndex: number; value: number }) => {
            const hour = params.dataIndex
            if (isServiceOff(hour)) return ''
            if (allLabels || hour === peakHour || hour === selectedHour) return formatNumber(params.value)
            return ''
          },
        },
        markArea: {
          silent: true,
          itemStyle: { color: 'rgba(255,255,255,0.035)' },
          label: { show: width >= 420, color: hex.text3, fontFamily: FONT, fontSize: 11, position: 'insideTop' },
          data: [[{ name: 'терминалы выключены', xAxis: '01' }, { xAxis: '04' }]],
        },
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { color: hex.border, type: 'dashed' },
          label: { color: hex.text3, fontFamily: FONT, fontSize: 11, formatter: (params: { value: number }) => formatNumber(params.value) },
          data: LOAD_THRESHOLDS.filter((threshold) => threshold < yMax).map((threshold) => ({ yAxis: threshold })),
        },
      },
    ],
  }
}

export interface PeriodChartInput {
  dates: readonly string[]
  totals: readonly (number | null)[]
  selectedDate: string
  yMax: number
  width: number
  period: 'week' | 'month'
}

export function periodChartOption({ dates, totals, selectedDate, yMax, width, period }: PeriodChartInput): ChartOption {
  const known = totals.map((total, index) => ({ total, index })).filter((item): item is { total: number; index: number } => item.total !== null)
  const max = known.reduce((best, item) => (item.total > best.total ? item : best), known[0] ?? { total: 0, index: -1 })
  const min = known.reduce((best, item) => (item.total < best.total ? item : best), known[0] ?? { total: 0, index: -1 })
  const allLabels = period === 'week' || width / dates.length >= 44
  return {
    textStyle: { fontFamily: FONT },
    grid: { left: 60, right: 16, top: 26, bottom: 30 },
    tooltip: {
      ...tooltip,
      formatter: (params: { dataIndex: number }[]) => {
        const index = params[0]?.dataIndex ?? 0
        const date = dates[index] ?? ''
        const total = totals[index]
        const body = total === null || total === undefined ? 'вне периода прогноза' : `${formatNumber(total)} посадок за сутки`
        return `${escape(formatShort(date))} — ${escape(body)}<br/><span style="color:${hex.text3}">щелчок — часы этого дня</span>`
      },
    },
    xAxis: {
      type: 'category',
      data: dates.map((date) => (period === 'week' ? formatShort(date) : String(Number(date.slice(8))))),
      ...axisBase(),
      axisLabel: {
        ...axisBase().axisLabel,
        interval: 0,
        formatter: (label: string, index: number) => {
          const date = dates[index] ?? ''
          if (date === selectedDate) return `{sel|${label}}`
          if (period === 'month' && width / dates.length < 22 && weekday(date) !== 1) return ''
          return weekday(date) >= 6 ? `{weekend|${label}}` : label
        },
        rich: {
          sel: { color: hex.text1, fontWeight: 600, fontFamily: FONT, fontSize: 12 },
          weekend: { color: hex.text2, fontFamily: FONT, fontSize: 12 },
        },
      },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: yMax,
      splitNumber: 4,
      ...axisBase(),
      axisLine: { show: false },
      axisLabel: { ...axisBase().axisLabel, formatter: (value: number) => formatNumber(value) },
      splitLine: { show: true, lineStyle: { color: hex.border, type: 'dashed', opacity: 0.6 } },
    },
    series: [
      {
        type: 'bar',
        barCategoryGap: period === 'week' ? '35%' : '20%',
        cursor: 'pointer',
        data: totals.map((total, index) => {
          const date = dates[index] ?? ''
          const selected = date === selectedDate
          return {
            value: total ?? 0,
            itemStyle: {
              color: inHorizon(date) ? hex.sum : 'transparent',
              borderRadius: [3, 3, 0, 0],
              borderColor: selected ? hex.selection : 'transparent',
              borderWidth: selected ? 2 : 0,
            },
          }
        }),
        label: {
          show: true,
          position: 'top',
          color: hex.text2,
          fontFamily: FONT,
          fontSize: 11,
          formatter: (params: { dataIndex: number; value: number }) => {
            const index = params.dataIndex
            if (totals[index] === null) return period === 'week' ? 'вне периода' : ''
            if (allLabels || index === max.index || index === min.index || dates[index] === selectedDate) return formatNumber(params.value)
            return ''
          },
        },
      },
    ],
  }
}
