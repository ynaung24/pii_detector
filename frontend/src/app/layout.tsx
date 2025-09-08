import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Toaster } from 'react-hot-toast'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'PII Detection Agent',
  description: 'Advanced PII detection using NER, Proximity Analysis, and Graph Theory',
  keywords: ['PII', 'Privacy', 'Data Protection', 'Security', 'Compliance'],
  authors: [{ name: 'PII Detection Team' }],
  viewport: 'width=device-width, initial-scale=1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gray-50`}>
        <div className="min-h-full">
          {/* Header */}
          <header className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center h-16">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="flex items-center">
                      <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <h1 className="text-xl font-semibold text-gray-900">
                          PII Detection Agent
                        </h1>
                        <p className="text-sm text-gray-500">
                          Advanced Privacy Protection
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <nav className="hidden md:flex space-x-8">
                  <a href="#" className="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">
                    Dashboard
                  </a>
                  <a href="#" className="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">
                    Reports
                  </a>
                  <a href="#" className="text-gray-500 hover:text-gray-700 px-3 py-2 text-sm font-medium">
                    Settings
                  </a>
                </nav>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
            {children}
          </main>

          {/* Footer */}
          <footer className="bg-white border-t border-gray-200 mt-auto">
            <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center">
                <div className="text-sm text-gray-500">
                  © 2024 PII Detection Agent. Built with Next.js and TypeScript.
                </div>
                <div className="flex space-x-4 text-sm text-gray-500">
                  <a href="#" className="hover:text-gray-700">Privacy Policy</a>
                  <a href="#" className="hover:text-gray-700">Terms of Service</a>
                  <a href="#" className="hover:text-gray-700">Documentation</a>
                </div>
              </div>
            </div>
          </footer>
        </div>
        
        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#22c55e',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />
      </body>
    </html>
  )
}
