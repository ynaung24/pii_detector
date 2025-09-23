'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { 
  Upload, 
  FileText, 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Download,
  Eye,
  EyeOff,
  BarChart3,
  Users,
  MapPin,
  Mail,
  Phone,
  CreditCard,
  User
} from 'lucide-react'
import toast from 'react-hot-toast'

interface PIIDetectionResult {
  success: boolean
  error?: string
  report_path?: string
  masked_csv_path?: string
  graph_visualization?: string
  graph_data?: string
  messages?: string[]
}

interface DetectionSummary {
  total_detections: number
  detection_types: Record<string, number>
  columns_affected: string[]
  rows_affected: number[]
}

interface ProximitySummary {
  total_analyses: number
  high_risk_rows: number
  medium_risk_rows: number
  low_risk_rows: number
}

interface GraphSummary {
  graph_stats: {
    nodes: number
    edges: number
    density: number
    average_clustering: number
  }
  reidentification_risk: {
    risk_level: 'low' | 'medium' | 'high'
    score: number
  }
}

interface PIIDetectionReport {
  metadata: {
    timestamp: string
    input_file: string
    proximity_window: number
  }
  ner_summary: DetectionSummary
  proximity_summary: ProximitySummary
  graph_summary: GraphSummary
  recommendations: string[]
}

export default function PIIDetectionAdvisor() {
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [detectionResult, setDetectionResult] = useState<PIIDetectionResult | null>(null)
  const [report, setReport] = useState<PIIDetectionReport | null>(null)
  const [showMaskedData, setShowMaskedData] = useState(false)
  const [activeView, setActiveView] = useState<'upload' | 'results' | 'report'>('upload')

  // Utility function to sanitize text content and prevent XSS
  const sanitizeText = (text: string): string => {
    if (typeof text !== 'string') return ''
    return text
      .replace(/[<>]/g, '') // Remove potential HTML tags
      .replace(/javascript:/gi, '') // Remove javascript: protocol
      .replace(/on\w+=/gi, '') // Remove event handlers
      .trim()
  }

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
        toast.error('Please upload a CSV file')
        return
      }
      if (file.size > 100 * 1024 * 1024) { // 100MB limit
        toast.error('File size must be less than 100MB')
        return
      }
      setUploadedFile(file)
      toast.success('File uploaded successfully')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    multiple: false
  })

  const processFile = async () => {
    if (!uploadedFile) return

    setIsProcessing(true)
    setActiveView('results')
    
    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)

      const response = await fetch('/api/process-csv', {
        method: 'POST',
        body: formData,
      })

      const result: PIIDetectionResult = await response.json()

      if (result.success) {
        setDetectionResult(result)
        
        // Fetch the detailed report
        if (result.report_path) {
          const reportResponse = await fetch('/api/report', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ report_path: result.report_path }),
          })
          const reportData = await reportResponse.json()
          setReport(reportData)
        }
        
        toast.success('PII detection completed successfully!')
        setActiveView('report')
      } else {
        toast.error(result.error || 'Processing failed')
        setActiveView('upload')
      }
    } catch (error) {
      console.error('Error processing file:', error)
      toast.error('Failed to process file. Please try again.')
      setActiveView('upload')
    } finally {
      setIsProcessing(false)
    }
  }

  const downloadFile = async (filePath: string, filename: string) => {
    try {
      // Sanitize filename to prevent XSS
      const sanitizedFilename = filename.replace(/[^a-zA-Z0-9._-]/g, '_')
      
      const response = await fetch('/api/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ file_path: filePath }),
      })

      if (response.ok) {
        const blob = await response.blob()
        
        // Use the safest download approach - no DOM manipulation
        if (window.navigator && (window.navigator as any).msSaveOrOpenBlob) {
          // Internet Explorer
          (window.navigator as any).msSaveOrOpenBlob(blob, sanitizedFilename)
        } else {
          // Modern browsers - use object URL with minimal DOM interaction
          const url = window.URL.createObjectURL(blob)
          
          // Create link element with minimal properties
          const link = document.createElement('a')
          link.setAttribute('href', url)
          link.setAttribute('download', sanitizedFilename)
          link.setAttribute('style', 'display: none; position: absolute; left: -9999px;')
          
          // Append to body temporarily
          document.body.appendChild(link)
          
          // Trigger download
          link.click()
          
          // Immediate cleanup
          setTimeout(() => {
            document.body.removeChild(link)
            window.URL.revokeObjectURL(url)
          }, 0)
        }
        toast.success(`${sanitizedFilename} downloaded successfully`)
      } else {
        toast.error('Failed to download file')
      }
    } catch (error) {
      console.error('Download error:', error)
      toast.error('Failed to download file')
    }
  }

  const getEntityIcon = (type: string) => {
    // Sanitize input to prevent XSS
    const sanitizedType = type.toLowerCase().replace(/[^a-zA-Z0-9_]/g, '')
    
    switch (sanitizedType) {
      case 'name':
        return <User className="w-4 h-4" />
      case 'email':
        return <Mail className="w-4 h-4" />
      case 'phone':
        return <Phone className="w-4 h-4" />
      case 'address':
      case 'location':
        return <MapPin className="w-4 h-4" />
      case 'credit_card':
        return <CreditCard className="w-4 h-4" />
      default:
        return <Shield className="w-4 h-4" />
    }
  }

  const getRiskBadge = (risk: string) => {
    // Sanitize risk level to prevent XSS
    const sanitizedRisk = risk.toLowerCase().replace(/[^a-zA-Z]/g, '')
    
    const riskClasses = {
      low: 'badge-success',
      medium: 'badge-warning',
      high: 'badge-danger'
    }
    return riskClasses[sanitizedRisk as keyof typeof riskClasses] || 'badge-info'
  }

  if (activeView === 'upload') {
    return (
      <div className="space-y-6">
        {/* Upload Section */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Upload CSV File</h3>
            <p className="mt-1 text-sm text-gray-500">
              Upload a CSV file to detect and analyze PII entities
            </p>
          </div>
          <div className="card-body">
            <div
              {...getRootProps()}
              className={`upload-area ${isDragActive ? 'dragover' : ''}`}
            >
              <input {...getInputProps()} />
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <div className="mt-4">
                <p className="text-lg font-medium text-gray-900">
                  {isDragActive ? 'Drop the file here' : 'Drag & drop a CSV file here'}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  or click to select a file
                </p>
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Maximum file size: 100MB
              </p>
            </div>

            {uploadedFile && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <FileText className="w-5 h-5 text-gray-400" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        {sanitizeText(uploadedFile.name)}
                      </p>
                      <p className="text-xs text-gray-500">
                        {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={processFile}
                    disabled={isProcessing}
                    className="btn-primary"
                  >
                    {isProcessing ? (
                      <>
                        <div className="loading-spinner mr-2"></div>
                        Processing...
                      </>
                    ) : (
                      <>
                        <Shield className="w-4 h-4 mr-2" />
                        Detect PII
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Features Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="card">
            <div className="card-body text-center">
              <Shield className="mx-auto h-8 w-8 text-primary-600 mb-3" />
              <h4 className="text-lg font-medium text-gray-900 mb-2">NER Detection</h4>
              <p className="text-sm text-gray-500">
                Advanced Named Entity Recognition using spaCy to identify names, emails, phones, and more.
              </p>
            </div>
          </div>

          <div className="card">
            <div className="card-body text-center">
              <BarChart3 className="mx-auto h-8 w-8 text-primary-600 mb-3" />
              <h4 className="text-lg font-medium text-gray-900 mb-2">Proximity Analysis</h4>
              <p className="text-sm text-gray-500">
                Context-based analysis to detect non-obvious PII and assess risk levels.
              </p>
            </div>
          </div>

          <div className="card">
            <div className="card-body text-center">
              <Users className="mx-auto h-8 w-8 text-primary-600 mb-3" />
              <h4 className="text-lg font-medium text-gray-900 mb-2">Graph Analysis</h4>
              <p className="text-sm text-gray-500">
                Entity relationship mapping to identify clusters and re-identification risks.
              </p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (activeView === 'results' && isProcessing) {
    return (
      <div className="card">
        <div className="card-body text-center py-12">
          <div className="loading-spinner mx-auto mb-4 w-8 h-8"></div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Processing Your File
          </h3>
          <p className="text-sm text-gray-500">
            Running NER detection, proximity analysis, and graph analysis...
          </p>
        </div>
      </div>
    )
  }

  if (activeView === 'report' && report) {
    return (
      <div className="space-y-6">
        {/* Report Header */}
        <div className="card">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-gray-900">PII Detection Report</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Generated on {new Date(report.metadata.timestamp).toLocaleString()}
                </p>
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={() => setActiveView('upload')}
                  className="btn-outline"
                >
                  New Analysis
                </button>
                {detectionResult?.report_path && (
                  <button
                    onClick={() => downloadFile(detectionResult.report_path!, 'pii_detection_report.json')}
                    className="btn-secondary"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Report
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="card">
            <div className="card-body">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <Shield className="h-8 w-8 text-primary-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Total Detections</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {report.ner_summary.total_detections}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-body">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <AlertTriangle className="h-8 w-8 text-danger-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">High Risk Rows</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {report.proximity_summary.high_risk_rows}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-body">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <BarChart3 className="h-8 w-8 text-warning-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Medium Risk Rows</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {report.proximity_summary.medium_risk_rows}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-body">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <CheckCircle className="h-8 w-8 text-success-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Low Risk Rows</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {report.proximity_summary.low_risk_rows}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Entity Types */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Detected Entity Types</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(report.ner_summary.detection_types).map(([type, count]) => (
                <div key={type} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                  {getEntityIcon(type)}
                  <div>
                    <p className="text-sm font-medium text-gray-900 capitalize">{sanitizeText(type)}</p>
                    <p className="text-xs text-gray-500">{count} detected</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Risk Assessment */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Risk Assessment</h3>
          </div>
          <div className="card-body">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-sm font-medium text-gray-900">Re-identification Risk</h4>
                <p className="text-sm text-gray-500">
                  Based on entity clustering and proximity analysis
                </p>
              </div>
              <span className={`badge ${getRiskBadge(report.graph_summary.reidentification_risk.risk_level)}`}>
                {report.graph_summary.reidentification_risk.risk_level.toUpperCase()}
              </span>
            </div>
            
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${report.graph_summary.reidentification_risk.score * 100}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Risk Score: {(report.graph_summary.reidentification_risk.score * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Recommendations */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Security Recommendations</h3>
          </div>
          <div className="card-body">
            <ul className="space-y-3">
              {report.recommendations.map((recommendation, index) => (
                <li key={index} className="flex items-start space-x-3">
                  <div className="flex-shrink-0">
                    <div className="w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center">
                      <span className="text-xs font-medium text-primary-600">{index + 1}</span>
                    </div>
                  </div>
                  <p className="text-sm text-gray-700">{sanitizeText(recommendation)}</p>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Download Options */}
        <div className="card">
          <div className="card-header">
            <h3 className="text-lg font-medium text-gray-900">Download Results</h3>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {detectionResult?.masked_csv_path && (
                <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <FileText className="w-5 h-5 text-gray-400" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">Masked CSV</p>
                      <p className="text-xs text-gray-500">PII replaced with ***</p>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setShowMaskedData(!showMaskedData)}
                      className="btn-outline text-xs"
                    >
                      {showMaskedData ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                    <button
                      onClick={() => downloadFile(detectionResult.masked_csv_path!, 'masked_data.csv')}
                      className="btn-primary text-xs"
                    >
                      <Download className="w-4 h-4 mr-1" />
                      Download
                    </button>
                  </div>
                </div>
              )}

              {detectionResult?.graph_visualization && (
                <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <BarChart3 className="w-5 h-5 text-gray-400" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">Graph Visualization</p>
                      <p className="text-xs text-gray-500">Entity relationship graph</p>
                    </div>
                  </div>
                  <button
                    onClick={() => downloadFile(detectionResult.graph_visualization!, 'pii_graph.png')}
                    className="btn-primary text-xs"
                  >
                    <Download className="w-4 h-4 mr-1" />
                    Download
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return null
}
