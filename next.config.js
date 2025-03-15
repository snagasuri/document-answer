const path = require('path');

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  poweredByHeader: false,
  webpack: (config, { isServer }) => {
    // Existing alias configuration
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': path.join(__dirname, 'src'),
    };

    // Add additional module resolution configuration
    config.resolve.modules = [
      path.join(__dirname, 'src'),
      'node_modules',
      ...config.resolve.modules || [],
    ];

    config.resolve.extensions = [
      '.ts', '.tsx', '.js', '.jsx', '.json',
      ...config.resolve.extensions || [],
    ];

    return config;
  },
};

module.exports = nextConfig;
