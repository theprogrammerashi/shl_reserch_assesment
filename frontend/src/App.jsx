import React, { useState } from 'react';
import axios from 'axios';
import { Search, Loader2, BookOpen, Clock, Globe, Briefcase } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = "http://localhost:8000";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setSearched(true);
    try {
      const res = await axios.post(`${API_URL}/recommend`, { query });
      setResults(res.data.recommended_assessments);
    } catch (err) {
      console.error(err);
      alert("Error fetching recommendations. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 text-gray-800 font-sans">

      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="bg-blue-600 text-white p-2 rounded-lg font-bold text-xl">SHL</div>
            <h1 className="text-xl font-bold text-gray-900 tracking-tight">Assessment Recommender</h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">

        {/* Search Hero */}
        <div className="text-center mb-12">
          <motion.h2
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-extrabold text-gray-900 mb-4"
          >
            Find the Perfect Assessment
          </motion.h2>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto"
          >
            Describe the role, skills, or job description, and our AI will recommend the best SHL solutions.
          </motion.p>

          <form onSubmit={handleSearch} className="max-w-2xl mx-auto relative">
            <input
              type="text"
              className="w-full pl-12 pr-4 py-4 rounded-xl border border-gray-200 shadow-lg focus:ring-4 focus:ring-blue-100 focus:border-blue-500 transition-all text-lg"
              placeholder="e.g. 'Senior Java Developer with strong leadership skills'"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={24} />
            <button
              type="submit"
              disabled={loading}
              className="absolute right-2 top-2 bottom-2 bg-blue-600 text-white px-6 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center"
            >
              {loading ? <Loader2 className="animate-spin" size={20} /> : "Search"}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {loading && (
            <div className="flex justify-center py-12">
              <Loader2 className="animate-spin text-blue-600" size={48} />
            </div>
          )}

          {!loading && searched && results.length === 0 && (
            <div className="text-center text-gray-500 py-12">
              No assessments found. Try a different query.
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
            <AnimatePresence>
              {results.map((item, index) => (
                <motion.div
                  key={item.url} // Use URL as key
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-white rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow p-6 flex flex-col h-full group"
                >
                  <div className="flex justify-between items-start mb-4">
                    <h3 className="text-xl font-bold text-gray-900 group-hover:text-blue-600 transition-colors line-clamp-2">
                      <a href={item.url} target="_blank" rel="noopener noreferrer">{item.name}</a>
                    </h3>
                    {item.adaptive_support === "Yes" && (
                      <span className="bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded-full font-medium ml-2 shrink-0">
                        Adaptive
                      </span>
                    )}
                  </div>

                  <p className="text-gray-600 mb-6 flex-grow line-clamp-3">
                    {item.description}
                  </p>

                  <div className="space-y-3 pt-4 border-t border-gray-50 text-sm text-gray-500">
                    <div className="flex items-center">
                      <Clock size={16} className="mr-2 text-gray-400" />
                      <span>{item.duration > 0 ? `${item.duration} mins` : "Duration varies"}</span>
                    </div>
                    <div className="flex items-center">
                      <Globe size={16} className="mr-2 text-gray-400" />
                      <span>{item.remote_support === "Yes" ? "Remote Ready" : "On-site"}</span>
                    </div>
                    {item.test_type && item.test_type.length > 0 && (
                      <div className="flex items-start">
                        <BookOpen size={16} className="mr-2 mt-1 text-gray-400" />
                        <div className="flex flex-wrap gap-1">
                          {item.test_type.map(type => (
                            <span key={type} className="bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded">
                              {type}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <a
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-6 w-full text-center bg-gray-50 hover:bg-gray-100 text-gray-900 py-2 rounded-lg font-medium transition-colors border border-gray-200"
                  >
                    View Details
                  </a>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
