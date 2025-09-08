'use client'

import { useState } from 'react'
import PIIDetectionAdvisor from '@/components/advisor'
import { Shield, FileText, BarChart3, AlertTriangle } from 'lucide-react'

export default function Home() {
  const [activeTab, setActiveTab] = useState('detection')

  const tabs = [
    { id: 'detection', label: 'PII Detection', icon: Shield },
    { id: 'reports', label: 'Reports', icon: FileText },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'alerts', label: 'Alerts', icon: AlertTriangle },
  ]

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-800 rounded-lg shadow-lg p-8 text-white">
        <div className="max-w-3xl">
          <h1 className="text-4xl font-bold mb-4">
            Advanced PII Detection & Privacy Protection
          </h1>
          <p className="text-xl text-primary-100 mb-6">
            Protect sensitive data with our comprehensive PII detection system using 
            Named Entity Recognition, Proximity Analysis, and Graph Theory.
          </p>
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-success-400 rounded-full"></div>
              <span className="text-sm">Real-time Detection</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-success-400 rounded-full"></div>
              <span className="text-sm">Risk Assessment</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-success-400 rounded-full"></div>
              <span className="text-sm">Compliance Ready</span>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 transition-colors duration-200`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            )
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="animate-fade-in">
        {activeTab === 'detection' && <PIIDetectionAdvisor />}
        {activeTab === 'reports' && <ReportsTab />}
        {activeTab === 'analytics' && <AnalyticsTab />}
        {activeTab === 'alerts' && <AlertsTab />}
      </div>
    </div>
  )
}

// Placeholder components for other tabs
function ReportsTab() {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-medium text-gray-900">Detection Reports</h3>
        <p className="mt-1 text-sm text-gray-500">
          View and manage your PII detection reports
        </p>
      </div>
      <div className="card-body">
        <div className="text-center py-12">
          <FileText className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No reports yet</h3>
          <p className="mt-1 text-sm text-gray-500">
            Start by uploading a CSV file to generate your first report.
          </p>
        </div>
      </div>
    </div>
  )
}

function AnalyticsTab() {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-medium text-gray-900">Analytics Dashboard</h3>
        <p className="mt-1 text-sm text-gray-500">
          Insights and trends from your PII detection data
        </p>
      </div>
      <div className="card-body">
        <div className="text-center py-12">
          <BarChart3 className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No analytics data</h3>
          <p className="mt-1 text-sm text-gray-500">
            Analytics will appear here after processing files.
          </p>
        </div>
      </div>
    </div>
  )
}

function AlertsTab() {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="text-lg font-medium text-gray-900">Security Alerts</h3>
        <p className="mt-1 text-sm text-gray-500">
          High-risk PII detections and security notifications
        </p>
      </div>
      <div className="card-body">
        <div className="text-center py-12">
          <AlertTriangle className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No alerts</h3>
          <p className="mt-1 text-sm text-gray-500">
            You'll be notified here when high-risk PII is detected.
          </p>
        </div>
      </div>
    </div>
  )
}
