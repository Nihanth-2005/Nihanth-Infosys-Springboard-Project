import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { signOut } from 'firebase/auth';
import { auth, db } from '../firebase/config';
import { collection, addDoc, query, where, getDocs, doc, deleteDoc } from 'firebase/firestore';
import axios from 'axios';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';
import { LogOut, Plus, FolderOpen, Sparkles, User, Trash2 } from 'lucide-react';
import { API_ENDPOINTS } from '../config/api';

function Dashboard({ user }) {
  const [workspaces, setWorkspaces] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [workspaceName, setWorkspaceName] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    loadWorkspaces();
  }, [user]);

  const loadWorkspaces = async () => {
    try {
      const q = query(collection(db, 'workspaces'), where('user_id', '==', user.uid));
      const querySnapshot = await getDocs(q);
      const firestoreWorkspaces = querySnapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));

      const response = await axios.post(API_ENDPOINTS.WORKSPACE_LIST, { user_id: user.uid });
      const backendWorkspaces = response.data.workspaces || [];

      const mergedWorkspaces = [...firestoreWorkspaces];
      backendWorkspaces.forEach(backendWs => {
        const exists = firestoreWorkspaces.some(fsWs => fsWs.workspace_id === backendWs.workspace_id);
        if (!exists) mergedWorkspaces.push(backendWs);
      });

      setWorkspaces(mergedWorkspaces);
    } catch (error) {
      console.error('Error loading workspaces:', error);
      toast.error('Failed to load workspaces');
    } finally {
      setLoading(false);
    }
  };

  const createWorkspace = async () => {
    if (!workspaceName.trim()) {
      toast.error('Please enter a workspace name');
      return;
    }

    try {
      const workspaceId = `ws_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      await addDoc(collection(db, 'workspaces'), {
        workspace_id: workspaceId,
        user_id: user.uid,
        name: workspaceName,
        created_at: new Date().toISOString()
      });

      await axios.post(API_ENDPOINTS.WORKSPACE_CREATE, {
        workspace_id: workspaceId,
        user_id: user.uid,
        name: workspaceName
      });

      toast.success('Workspace created successfully!');
      setShowCreateModal(false);
      setWorkspaceName('');
      loadWorkspaces();
      navigate(`/workspace/${workspaceId}`);
    } catch (error) {
      console.error('Error creating workspace:', error);
      toast.error('Failed to create workspace');
    }
  };

  const handleLogout = async () => {
    try {
      await signOut(auth);
      toast.success('Logged out successfully');
    } catch (error) {
      toast.error(error.message);
    }
  };

  const deleteWorkspace = async (workspace) => {
    const workspaceId = workspace.workspace_id || workspace.id;
    const workspaceName = workspace.name;

    if (!window.confirm(`Delete "${workspaceName}"? This action is irreversible.`)) return;

    try {
      // First delete from backend (SQLite)
      await axios.post(API_ENDPOINTS.WORKSPACE_DELETE, {
        workspace_id: workspaceId,
        user_id: user.uid
      });

      // Then delete from Firestore if it exists
      if (workspace.id) {
        await deleteDoc(doc(db, 'workspaces', workspace.id));
      }

      toast.success('Workspace deleted successfully');
      loadWorkspaces();
    } catch (error) {
      console.error('Error deleting workspace:', error);
      const errorMessage = error.response?.data?.error || 'Failed to delete workspace';
      toast.error(errorMessage);

      // If backend deletion failed, don't proceed with Firestore deletion
      if (error.response?.status === 404) {
        toast.error('Workspace not found. It may have already been deleted.');
        // Still try to delete from Firestore in case it exists there
        if (workspace.id) {
          try {
            await deleteDoc(doc(db, 'workspaces', workspace.id));
            toast.success('Workspace removed from local storage');
            loadWorkspaces();
          } catch (firestoreError) {
            console.error('Firestore deletion also failed:', firestoreError);
          }
        }
      }
    }
  };

  return (
    <div
  className="min-h-screen relative overflow-hidden text-slate-800 dark:text-slate-100"
  style={{
    backgroundImage: 'url(https://img.freepik.com/free-vector/technology-network-background-connection-digital-data-visualization_1017-27428.jpg)',
    backgroundSize: 'cover',
    backgroundPosition: 'center 40%',
    backgroundAttachment: 'fixed',
  }}
>

      {/* Subtle overlay for contrast */}
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px] pointer-events-none"></div>

      {/* Header */}
      <header className="glass-card border-b border-slate-200 dark:border-slate-700 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold gradient-text">AI Workspace</h1>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-slate-200 dark:text-slate-300">
                <User className="w-5 h-5" />
                <span className="text-sm">{user.email}</span>
              </div>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </motion.button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">Your Workspaces</h2>
          <p className="text-slate-300">Create and manage your AI workspaces</p>
        </div>

        {/* Create Workspace Button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setShowCreateModal(true)}
          className="mb-8 w-full sm:w-auto flex items-center gap-3 px-6 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all"
        >
          <Plus className="w-5 h-5" />
          <span>Create New Workspace</span>
        </motion.button>

        {/* Workspaces Grid */}
        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          </div>
        ) : workspaces.length === 0 ? (
          <div className="text-center py-12 glass-card rounded-2xl bg-gradient-to-br from-pink-100 via-purple-100 to-purple-200 border border-purple-200 shadow-md">
  <FolderOpen className="w-16 h-16 mx-auto text-purple-500 mb-4" />
  <h3 className="text-xl font-semibold text-black mb-2">
    No workspaces yet
  </h3>
  <p className="text-black/80 mb-4">
    Create your first workspace to get started
  </p>
</div>

        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {workspaces.map((workspace, index) => (
              <motion.div
                key={workspace.workspace_id || workspace.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="rounded-xl p-6 bg-gradient-to-br from-pink-200 via-purple-300 to-purple-400 text-slate-900 border border-purple-300 shadow-md hover:shadow-purple-400/50 hover:scale-[1.02] transition-all duration-300"



              >
                <div className="flex items-start justify-between mb-4">
                  <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                    <FolderOpen className="w-6 h-6 text-white" />
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteWorkspace(workspace);
                    }}
                    className="p-2 text-red-500 hover:text-red-700 hover:bg-red-50/20 rounded-lg transition-colors"
                    title="Delete workspace"
                  >
                    <Trash2 className="w-4 h-4" />
                  </motion.button>
                </div>
                <div
                  onClick={() => navigate(`/workspace/${workspace.workspace_id || workspace.id}`)}
                  className="cursor-pointer"
                >
                  <h3 className="text-xl font-semibold text-black mb-2">
  {workspace.name}
</h3>
<p className="text-sm text-black/80">
  Created {new Date(workspace.created_at).toLocaleDateString()}
</p>

                </div>
              </motion.div>
            ))}
          </div>
        )}
      </main>

      {/* Create Workspace Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="rounded-xl p-6 bg-gradient-to-br from-blue-600 to-indigo-700 text-white shadow-lg hover:shadow-xl transition-all hover:scale-[1.02] hover:shadow-blue-500/40 border border-blue-500/30"



          >
            <h3 className="text-2xl font-bold mb-4 text-white">Create New Workspace</h3>
            <input
  type="text"
  value={workspaceName}
  onChange={(e) => setWorkspaceName(e.target.value)}
  placeholder="Workspace name"
  className="w-full px-4 py-3 rounded-lg border border-slate-300 bg-white text-black placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 mb-4"
  onKeyPress={(e) => e.key === 'Enter' && createWorkspace()}
/>

            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setWorkspaceName('');
                }}
                className="flex-1 px-4 py-2 rounded-lg border border-slate-300 hover:bg-slate-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={createWorkspace}
                className="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:shadow-lg transition-all"
              >
                Create
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
