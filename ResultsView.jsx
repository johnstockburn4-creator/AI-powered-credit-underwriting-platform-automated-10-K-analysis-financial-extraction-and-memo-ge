import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import MetricsPanel from './MetricsPanel'
import ReviewCorrect from './ReviewCorrect'

const TABS = [
  { id: 'memo',    label: 'Credit Memo' },
  { id: 'mda',     label: 'MD&A Analysis' },
  { id: 'metrics', label: 'Extracted Metrics' },
  { id: 'review',  label: 'Review & Correct' },
]

export default function ResultsView({ data, companyName, covenants }) {
  const [tab, setTab] = useState('memo')

  const completeness  = Math.round((data.completeness || 0) * 100)
  const flags         = data.validation_flags || []
  const profileFlags  = flags.filter(f => f.startsWith('PROFILE'))
  const otherFlags    = flags.filter(f => !f.startsWith('PROFILE'))
  const periods       = (data.extracted?.periods || []).map(p => p.period_name)

  return (
    <div>
      {/* Summary bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '1.5rem',
        padding: '0.75rem 1rem', background: 'var(--surface)',
        border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)',
        marginBottom: '1rem', flexWrap: 'wrap',
      }}>
        <Pill label="Completeness" value={`${completeness}%`}
          color={completeness >= 80 ? 'var(--green)' : completeness >= 55 ? 'var(--accent)' : 'var(--red)'} />
        <Pill label="Periods" value={periods.join(', ') || '—'} color="var(--blue)" />
        <Pill label="Basis"   value={data.extracted?.statement_basis || '—'} color="var(--text2)" />
        {otherFlags.length > 0   && <Pill label="Flags"            value={otherFlags.length}   color="var(--red)"    />}
        {profileFlags.length > 0 && <Pill label="Profile Conflicts" value={profileFlags.length} color="var(--accent)" />}
        <div style={{ marginLeft: 'auto', color: 'var(--text3)', fontSize: '0.78rem', fontFamily: 'DM Mono, monospace' }}>
          {data.run_id?.slice(0, 8)}
        </div>
      </div>

      {/* Export bar */}
      {data.run_id && (
        <ExportBar
          runId={data.run_id}
          borrowerName={companyName}
          covenants={covenants}
        />
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', marginBottom: '1.5rem' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            background: 'none', border: 'none', padding: '0.6rem 1.25rem',
            fontSize: '0.875rem', fontFamily: 'inherit', cursor: 'pointer',
            color: tab === t.id ? 'var(--accent)' : 'var(--text3)',
            borderBottom: tab === t.id ? '2px solid var(--accent)' : '2px solid transparent',
            marginBottom: -1, fontWeight: tab === t.id ? 600 : 400,
            transition: 'all 0.15s', position: 'relative',
          }}>
            {t.label}
            {t.id === 'review' && profileFlags.length > 0 && (
              <span style={{
                position: 'absolute', top: 6, right: 6,
                width: 7, height: 7, borderRadius: '50%',
                background: 'var(--accent)',
              }} />
            )}
          </button>
        ))}
      </div>

      {tab === 'memo'    && <MarkdownPane content={data.memo_markdown} copyLabel="Copy Memo" />}
      {tab === 'mda'     && <MarkdownPane content={data.mda_summary}   copyLabel="Copy MD&A" />}
      {tab === 'metrics' && <MetricsPanel extracted={data.extracted} />}
      {tab === 'review'  && <ReviewCorrect data={data} companyName={companyName} />}

      {/* Non-profile flags */}
      {otherFlags.length > 0 && (
        <div style={{
          marginTop: '1.5rem', background: 'rgba(224,92,92,0.06)',
          border: '1px solid rgba(224,92,92,0.2)', borderRadius: 'var(--radius)', padding: '1rem',
        }}>
          <p style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--red)',
            textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.5rem' }}>
            Validation Flags
          </p>
          {otherFlags.map((f, i) => (
            <p key={i} style={{ fontSize: '0.85rem', color: 'var(--text2)', marginBottom: '0.2rem' }}>⚠ {f}</p>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Export bar ────────────────────────────────────────────────────────────────

function ExportBar({ runId, borrowerName, covenants }) {
  const [excelState, setExcelState] = useState('idle')  // idle | loading | error
  const [wordState,  setWordState]  = useState('idle')
  const [errorMsg,   setErrorMsg]   = useState('')

  const triggerDownload = (blob, filename) => {
    const url = URL.createObjectURL(blob)
    const a   = document.createElement('a')
    a.href = url; a.download = filename
    document.body.appendChild(a); a.click()
    document.body.removeChild(a); URL.revokeObjectURL(url)
  }

  const exportFile = async (type) => {
    const setState  = type === 'excel' ? setExcelState : setWordState
    const endpoint  = type === 'excel' ? '/v1/export/docs/excel' : '/v1/export/docs/word'
    const ext       = type === 'excel' ? 'xlsx' : 'docx'
    const suffix    = type === 'excel' ? 'summary' : 'memo'

    setState('loading')
    setErrorMsg('')

    try {
      const body = new FormData()
      body.append('run_id', runId)
      if (borrowerName) body.append('borrower_name', borrowerName)
      if (type === 'word' && covenants) {
        body.append('covenants_json', JSON.stringify(covenants))
      }

      const res = await fetch(`http://localhost:8000${endpoint}`, { method: 'POST', body })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Export failed' }))
        throw new Error(err.detail || 'Export failed')
      }

      const blob     = await res.blob()
      const safeName = (borrowerName || 'Company').replace(/\s+/g, '_').replace(/\//g, '-')
      triggerDownload(blob, `${safeName}_credit_${suffix}.${ext}`)
      setState('idle')

    } catch (e) {
      setState('error')
      setErrorMsg(e.message)
    }
  }

  const btnBase = {
    display: 'inline-flex', alignItems: 'center', gap: '0.4rem',
    padding: '0.4rem 0.85rem', borderRadius: 'var(--radius)',
    fontSize: '0.8rem', fontFamily: 'inherit', fontWeight: 500,
    cursor: 'pointer', transition: 'opacity 0.15s',
    border: '1px solid var(--border)',
  }

  const isLoading = excelState === 'loading' || wordState === 'loading'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '0.6rem',
      padding: '0.6rem 0.75rem',
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius)', marginBottom: '1rem',
    }}>
      <span style={{
        fontSize: '0.7rem', fontWeight: 700, color: 'var(--text3)',
        textTransform: 'uppercase', letterSpacing: '0.06em', marginRight: '0.25rem',
      }}>
        Export
      </span>

      {/* Excel */}
      <button
        disabled={isLoading}
        onClick={() => exportFile('excel')}
        style={{
          ...btnBase,
          background: excelState === 'loading' ? 'var(--surface2)' : 'rgba(76,175,130,0.1)',
          color: excelState === 'error' ? 'var(--red)' : 'var(--green)',
          borderColor: excelState === 'error' ? 'rgba(224,92,92,0.35)' : 'rgba(76,175,130,0.3)',
          opacity: isLoading ? 0.6 : 1,
          cursor: isLoading ? 'not-allowed' : 'pointer',
        }}
      >
        {excelState === 'loading' ? <Spinner /> : '📊'}
        {excelState === 'loading' ? 'Building…' : excelState === 'error' ? 'Failed — retry' : 'Excel'}
      </button>

      {/* Word */}
      <button
        disabled={isLoading}
        onClick={() => exportFile('word')}
        style={{
          ...btnBase,
          background: wordState === 'loading' ? 'var(--surface2)' : 'rgba(91,141,232,0.1)',
          color: wordState === 'error' ? 'var(--red)' : 'var(--blue)',
          borderColor: wordState === 'error' ? 'rgba(224,92,92,0.35)' : 'rgba(91,141,232,0.3)',
          opacity: isLoading ? 0.6 : 1,
          cursor: isLoading ? 'not-allowed' : 'pointer',
        }}
      >
        {wordState === 'loading' ? <Spinner /> : '📄'}
        {wordState === 'loading' ? 'Building… (30–60s)' : wordState === 'error' ? 'Failed — retry' : 'Word Doc'}
      </button>

      {/* Inline status / error */}
      {wordState === 'loading' && (
        <span style={{ fontSize: '0.75rem', color: 'var(--text3)', fontStyle: 'italic' }}>
          Parsing segments from MD&A…
        </span>
      )}
      {errorMsg && (
        <span style={{
          fontSize: '0.75rem', color: 'var(--red)',
          background: 'rgba(224,92,92,0.08)', padding: '0.2rem 0.5rem',
          borderRadius: 4, border: '1px solid rgba(224,92,92,0.2)',
        }}>
          {errorMsg}
        </span>
      )}
    </div>
  )
}

// ── Shared helpers ────────────────────────────────────────────────────────────

function Spinner() {
  return (
    <span style={{
      display: 'inline-block', width: 11, height: 11,
      border: '2px solid currentColor', borderTopColor: 'transparent',
      borderRadius: '50%', animation: 'spin 0.7s linear infinite',
    }} />
  )
}

function Pill({ label, value, color }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
      <span style={{ fontSize: '0.75rem', color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</span>
      <span style={{ fontSize: '0.8rem', fontWeight: 600, color, fontFamily: 'DM Mono, monospace' }}>{value}</span>
    </div>
  )
}

function MarkdownPane({ content, copyLabel }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    navigator.clipboard.writeText(content || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  if (!content || content.startsWith('Error')) {
    return (
      <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text3)' }}>
        {content?.startsWith('Error')
          ? <p style={{ color: 'var(--red)', fontSize: '0.9rem' }}>{content}</p>
          : <p>No content available.</p>}
      </div>
    )
  }
  return (
    <div style={{ position: 'relative' }}>
      <button onClick={handleCopy} style={{
        position: 'absolute', top: 0, right: 0, zIndex: 2,
        background: 'var(--surface2)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius)', padding: '0.35rem 0.75rem',
        fontSize: '0.78rem', color: 'var(--text3)', fontFamily: 'inherit', cursor: 'pointer',
      }}>
        {copied ? '✓ Copied' : copyLabel}
      </button>
      <div className="markdown" style={{
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)', padding: '2rem', maxHeight: '75vh', overflowY: 'auto',
      }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  )
}
